import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from UniADet_lib import (
    UniADetZeroShotLateFusion,
    available_models,
    get_backbone_data_config,
    load_backbone,
)
from uniadet_dataset import UniADetDataset
from utils.backbone_config import resolve_features_list
from utils.logger import get_logger
from utils.loss import BinaryDiceLoss, FocalLoss
from utils.transforms import get_transform
from utils.uniadet_eval import evaluate_uniadet_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def compute_segmentation_loss(seg_probs_per_layer, gt_mask, focal_loss, dice_loss):
    gt_mask = (gt_mask > 0.5).float()
    gt_mask_2d = gt_mask.squeeze(1)

    seg_losses = []
    for seg_probs in seg_probs_per_layer:
        seg_losses.append(focal_loss(seg_probs, gt_mask))
        seg_losses.append(dice_loss(seg_probs[:, 1, :, :], gt_mask_2d))

    return sum(seg_losses) / max(len(seg_losses), 1)


def save_checkpoint(model, args, epoch, checkpoint_path, extra_fields=None):
    checkpoint = {
        "state_dict": model.extra_state_dict(),
        "backbone": args.backbone,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "temperature": args.temperature,
        "score_fusion_weight": args.score_fusion_weight,
        "cls_loss_weight": args.cls_loss_weight,
        "seg_loss_weight": args.seg_loss_weight,
        "epoch": epoch,
        "fusion_mode": "late_fusion",
    }
    if extra_fields:
        checkpoint.update(extra_fields)
    torch.save(checkpoint, checkpoint_path)


def select_eval_metric(metric_summary, metric_name):
    mean_metrics = metric_summary.get("mean", {})
    if not mean_metrics:
        return None

    if metric_name == "image_auc":
        return mean_metrics.get("sample_auroc")
    if metric_name == "pixel_auc":
        return mean_metrics.get("pixel_auroc")
    if metric_name == "mean_auc":
        sample_auroc = mean_metrics.get("sample_auroc")
        pixel_auroc = mean_metrics.get("pixel_auroc")
        if sample_auroc is None or pixel_auroc is None:
            return None
        return 0.5 * (sample_auroc + pixel_auroc)

    raise ValueError(f"Unsupported eval metric: {metric_name}")


def save_eval_history(save_path, eval_history):
    history_path = os.path.join(save_path, "eval_history.json")
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(eval_history, fp, indent=2)


def train(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    backbone_cfg = get_backbone_data_config(args.backbone, image_size=args.image_size)
    args.image_size = backbone_cfg["image_size"]
    backbone = load_backbone(args.backbone, device=device, image_size=args.image_size)
    args.features_list = resolve_features_list(args.features_list, backbone.total_layers, logger=logger)

    model = UniADetZeroShotLateFusion(
        backbone=backbone,
        feature_layers=args.features_list,
        image_size=args.image_size,
        temperature=args.temperature,
        score_fusion_weight=args.score_fusion_weight,
    ).to(device)

    preprocess, target_transform = get_transform(args, mean=backbone_cfg["mean"], std=backbone_cfg["std"])
    train_data = UniADetDataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.train_dataset,
        mode=args.data_mode,
        enable_caa=args.enable_caa,
        caa_prob=args.caa_prob,
        caa_grid_sizes=tuple(args.caa_grid_sizes),
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=args.learning_rate * 0.1,
    )
    focal_loss = FocalLoss()
    dice_loss = BinaryDiceLoss()
    periodic_eval_enabled = bool(args.test_data_path and args.test_dataset and args.eval_interval > 0)
    best_eval_score = float("-inf")
    best_eval_epoch = None
    eval_history = []

    logger.info(
        f"UniADet-ZS-LateFusion | Train: {args.train_dataset} ({args.data_mode}) | Backbone: {args.backbone} | "
        f"Image: {args.image_size} | Layers: {args.features_list} | CAA: {args.enable_caa} | "
        f"Mean/Std: {backbone_cfg['mean']} / {backbone_cfg['std']} | "
        f"Loss Weights: cls={args.cls_loss_weight}, seg={args.seg_loss_weight}"
    )
    if periodic_eval_enabled:
        logger.info(
            f"Periodic evaluation enabled: every {args.eval_interval} epochs on "
            f"{args.test_dataset} ({args.eval_data_mode}), selecting best by {args.eval_metric}."
        )
    else:
        logger.info(
            "Periodic evaluation disabled. Provide --test_data_path, --test_dataset, "
            "and --eval_interval > 0 to enable best-checkpoint selection."
        )

    for epoch in range(args.epoch):
        model.train()
        cls_loss_meter = []
        seg_loss_meter = []
        total_loss_meter = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for batch in progress:
            image = batch["img"].to(device)
            label = batch["anomaly"].to(device).long()
            gt_mask = batch["img_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)

            cls_losses = [
                F.cross_entropy(cls_logits, label)
                for cls_logits in outputs["cls_logits_per_layer"]
            ]
            cls_loss = sum(cls_losses) / max(len(cls_losses), 1)
            seg_loss = compute_segmentation_loss(outputs["seg_probs_per_layer"], gt_mask, focal_loss, dice_loss)
            total_loss = args.cls_loss_weight * cls_loss + args.seg_loss_weight * seg_loss

            total_loss.backward()
            optimizer.step()

            cls_loss_meter.append(float(cls_loss.item()))
            seg_loss_meter.append(float(seg_loss.item()))
            total_loss_meter.append(float(total_loss.item()))
            progress.set_postfix(
                cls=f"{np.mean(cls_loss_meter):.4f}",
                seg=f"{np.mean(seg_loss_meter):.4f}",
                total=f"{np.mean(total_loss_meter):.4f}",
            )

        scheduler.step()
        logger.info(
            f"Epoch [{epoch + 1}/{args.epoch}] - cls: {np.mean(cls_loss_meter):.4f}, "
            f"seg: {np.mean(seg_loss_meter):.4f}, total: {np.mean(total_loss_meter):.4f}"
        )

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            save_checkpoint(model, args, epoch + 1, checkpoint_path)

        should_eval = periodic_eval_enabled and (
            (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epoch
        )
        if should_eval:
            logger.info(f"Running evaluation at epoch {epoch + 1} on {args.test_dataset}...")
            model.eval()
            metric_summary = evaluate_uniadet_model(
                model=model,
                dataset_root=args.test_data_path,
                dataset_name=args.test_dataset,
                preprocess=preprocess,
                target_transform=target_transform,
                device=device,
                logger=logger,
                data_mode=args.eval_data_mode,
                num_workers=args.num_workers,
                desc=f"UniADet-ZS-LateFusion eval @ epoch {epoch + 1}",
                compute_original_size_metrics=args.eval_original_size,
            )
            selected_score = select_eval_metric(metric_summary, args.eval_metric)
            mean_metrics = metric_summary.get("mean", {})
            original_size_mean_metrics = metric_summary.get("original_size", {}).get("mean", {})
            history_entry = {
                "epoch": epoch + 1,
                "selection_metric": args.eval_metric,
                "selection_score": round(selected_score, 6) if selected_score is not None else None,
            }
            history_entry.update({key: round(value, 6) for key, value in mean_metrics.items()})
            history_entry.update(
                {
                    f"orig_{key}": round(value, 6)
                    for key, value in original_size_mean_metrics.items()
                }
            )
            eval_history.append(history_entry)
            save_eval_history(args.save_path, eval_history)
            logger.info(
                "Epoch %d eval summary | Image-AUC: %.4f | Pixel-AUC: %.4f | %s: %.4f",
                epoch + 1,
                mean_metrics.get("sample_auroc", 0.0),
                mean_metrics.get("pixel_auroc", 0.0),
                args.eval_metric,
                selected_score if selected_score is not None else 0.0,
            )
            if original_size_mean_metrics:
                logger.info(
                    "Epoch %d original-size eval | Image-AUC: %.4f | Pixel-AUC: %.4f",
                    epoch + 1,
                    original_size_mean_metrics.get("sample_auroc", 0.0),
                    original_size_mean_metrics.get("pixel_auroc", 0.0),
                )

            if selected_score is not None and selected_score > best_eval_score:
                best_eval_score = selected_score
                best_eval_epoch = epoch + 1
                best_checkpoint_path = os.path.join(args.save_path, "best_model.pth")
                save_checkpoint(
                    model,
                    args,
                    epoch + 1,
                    best_checkpoint_path,
                    extra_fields={
                        "best_metric_name": args.eval_metric,
                        "best_metric_value": float(selected_score),
                        "best_metric_epoch": epoch + 1,
                        "eval_summary": mean_metrics,
                        "original_size_eval_summary": original_size_mean_metrics,
                    },
                )
                logger.info(
                    "New best checkpoint saved to %s (epoch %d, %s=%.4f)",
                    best_checkpoint_path,
                    epoch + 1,
                    args.eval_metric,
                    selected_score,
                )

    final_path = os.path.join(args.save_path, "final_model.pth")
    save_checkpoint(model, args, args.epoch, final_path)
    logger.info(f"Training completed. Final checkpoint saved to {final_path}")
    if best_eval_epoch is not None:
        logger.info(
            "Best checkpoint summary | epoch: %d | %s: %.4f",
            best_eval_epoch,
            args.eval_metric,
            best_eval_score,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UniADet Zero-Shot Training (Late Fusion)", add_help=True)
    parser.add_argument("--train_data_path", type=str, required=True, help="auxiliary training dataset path")
    parser.add_argument("--save_path", type=str, default="./checkpoints_uniadet_zs_late_fusion", help="checkpoint save path")
    parser.add_argument("--train_dataset", type=str, default="visa", help="training dataset name")
    parser.add_argument("--data_mode", type=str, default="test", choices=["train", "test"], help="meta split to use")
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-L/14@336px",
        choices=available_models(),
        help="backbone to use",
    )
    parser.add_argument("--features_list", type=int, nargs="*", default=[12, 15, 18, 21, 24], help="feature layers")
    parser.add_argument("--epoch", type=int, default=15, help="training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=None, help="image size (defaults to backbone config)")
    parser.add_argument("--temperature", type=float, default=0.07, help="cosine similarity temperature")
    parser.add_argument("--score_fusion_weight", type=float, default=0.5, help="lambda_p in zero-shot inference")
    parser.add_argument("--cls_loss_weight", type=float, default=1.0, help="weight for classification loss")
    parser.add_argument("--seg_loss_weight", type=float, default=1.0, help="weight for segmentation loss")
    parser.add_argument("--enable_caa", action="store_true", help="enable class-aware augmentation")
    parser.add_argument("--caa_prob", type=float, default=0.5, help="probability of applying CAA")
    parser.add_argument("--caa_grid_sizes", type=int, nargs="*", default=[2, 3], help="CAA grid sizes")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--save_freq", type=int, default=1, help="checkpoint save frequency")
    parser.add_argument("--test_data_path", type=str, default=None, help="optional evaluation dataset path")
    parser.add_argument("--test_dataset", type=str, default=None, help="optional evaluation dataset name")
    parser.add_argument("--eval_data_mode", type=str, default="test", choices=["train", "test"], help="evaluation split")
    parser.add_argument("--eval_interval", type=int, default=3, help="run evaluation every N epochs; <=0 disables it")
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="mean_auc",
        choices=["mean_auc", "image_auc", "pixel_auc"],
        help="metric used to select best checkpoint",
    )
    parser.add_argument("--eval_original_size", action="store_true", help="also compute original-size mask metrics during eval")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
