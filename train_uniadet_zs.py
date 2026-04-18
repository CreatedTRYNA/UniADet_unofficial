import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from UniADet_lib import UniADetZeroShot, available_models, get_backbone_data_config, load_backbone
from uniadet_dataset import UniADetDataset
from utils.backbone_config import resolve_features_list
from utils.logger import get_logger
from utils.loss import BinaryDiceLoss, FocalLoss
from utils.transforms import get_transform


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


def save_checkpoint(model, args, epoch, checkpoint_path):
    checkpoint = {
        "state_dict": model.extra_state_dict(),
        "backbone": args.backbone,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "temperature": args.temperature,
        "score_fusion_weight": args.score_fusion_weight,
        "epoch": epoch,
    }
    torch.save(checkpoint, checkpoint_path)


def train(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    backbone_cfg = get_backbone_data_config(args.backbone, image_size=args.image_size)
    args.image_size = backbone_cfg["image_size"]
    backbone = load_backbone(args.backbone, device=device, image_size=args.image_size)
    args.features_list = resolve_features_list(args.features_list, backbone.total_layers, logger=logger)

    model = UniADetZeroShot(
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

    logger.info(
        f"UniADet-ZS | Train: {args.train_dataset} ({args.data_mode}) | Backbone: {args.backbone} | "
        f"Image: {args.image_size} | Layers: {args.features_list} | CAA: {args.enable_caa} | "
        f"Mean/Std: {backbone_cfg['mean']} / {backbone_cfg['std']}"
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
            total_loss = cls_loss + seg_loss

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

    final_path = os.path.join(args.save_path, "final_model.pth")
    save_checkpoint(model, args, args.epoch, final_path)
    logger.info(f"Training completed. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UniADet Zero-Shot Training", add_help=True)
    parser.add_argument("--train_data_path", type=str, required=True, help="auxiliary training dataset path")
    parser.add_argument("--save_path", type=str, default="./checkpoints_uniadet_zs", help="checkpoint save path")
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
    parser.add_argument("--enable_caa", action="store_true", help="enable class-aware augmentation")
    parser.add_argument("--caa_prob", type=float, default=0.5, help="probability of applying CAA")
    parser.add_argument("--caa_grid_sizes", type=int, nargs="*", default=[2, 3], help="CAA grid sizes")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--save_freq", type=int, default=1, help="checkpoint save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
