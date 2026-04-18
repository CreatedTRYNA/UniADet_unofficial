import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from UniADet_lib import UniADetZeroShot, get_backbone_data_config, load_backbone
from uniadet_dataset import UniADetDataset
from utils.logger import get_logger
from utils.metrics import compute_metrics
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


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = load_backbone(
        checkpoint["backbone"],
        device=device,
        image_size=checkpoint["image_size"],
    )
    model = UniADetZeroShot(
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

    test_data = UniADetDataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.test_dataset,
        mode=args.data_mode,
        enable_caa=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    results = {
        obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}
        for obj in test_data.obj_list
    }

    for batch in tqdm(test_loader, desc="UniADet-ZS test"):
        image = batch["img"].to(device)
        cls_name = batch["cls_name"][0]
        gt_mask = (batch["img_mask"] > 0.5).float()
        anomaly_label = int(batch["anomaly"].item())

        with torch.no_grad():
            outputs = model(image)
            anomaly_map = outputs["anomaly_map"].cpu()
            image_score = outputs["image_score"].cpu()

        results[cls_name]["gt_sp"].append(anomaly_label)
        results[cls_name]["pr_sp"].append(float(image_score.item()))
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["anomaly_maps"].append(anomaly_map)

    compute_metrics(results, test_data.obj_list, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UniADet Zero-Shot Test", add_help=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="test dataset path")
    parser.add_argument("--save_path", type=str, default="./test_results_uniadet_zs", help="result save path")
    parser.add_argument("--test_dataset", type=str, required=True, help="test dataset name")
    parser.add_argument("--data_mode", type=str, default="test", choices=["train", "test"], help="meta split to use")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
