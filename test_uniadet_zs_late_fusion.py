import argparse
import os
import random

import numpy as np
import torch

from UniADet_lib import UniADetZeroShotLateFusion, get_backbone_data_config, load_backbone
from utils.logger import get_logger
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


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = load_backbone(
        checkpoint["backbone"],
        device=device,
        image_size=checkpoint["image_size"],
    )
    model = UniADetZeroShotLateFusion(
        backbone=backbone,
        feature_layers=checkpoint["features_list"],
        image_size=checkpoint["image_size"],
        temperature=checkpoint.get("temperature", 0.07),
        score_fusion_weight=checkpoint.get("score_fusion_weight", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint_path, device)
    args.image_size = checkpoint["image_size"]
    backbone_cfg = get_backbone_data_config(checkpoint["backbone"], image_size=args.image_size)
    preprocess, target_transform = get_transform(args, mean=backbone_cfg["mean"], std=backbone_cfg["std"])

    evaluate_uniadet_model(
        model=model,
        dataset_root=args.test_data_path,
        dataset_name=args.test_dataset,
        preprocess=preprocess,
        target_transform=target_transform,
        device=device,
        logger=logger,
        data_mode=args.data_mode,
        num_workers=args.num_workers,
        desc="UniADet-ZS-LateFusion test",
        compute_original_size_metrics=args.eval_original_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UniADet Zero-Shot Test (Late Fusion)", add_help=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="test dataset path")
    parser.add_argument("--save_path", type=str, default="./test_results_uniadet_zs_late_fusion", help="result save path")
    parser.add_argument("--test_dataset", type=str, required=True, help="test dataset name")
    parser.add_argument("--data_mode", type=str, default="test", choices=["train", "test"], help="meta split to use")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--eval_original_size", action="store_true", help="also compute original-size mask metrics")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
