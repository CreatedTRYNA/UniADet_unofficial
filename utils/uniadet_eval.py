import torch
import torch.nn.functional as F
from tqdm import tqdm

from uniadet_dataset import UniADetDataset
from utils.metrics import compute_metrics


def evaluate_uniadet_model(
    model,
    dataset_root,
    dataset_name,
    preprocess,
    target_transform,
    device,
    logger,
    data_mode="test",
    num_workers=0,
    desc="UniADet eval",
    compute_original_size_metrics=False,
):
    eval_data = UniADetDataset(
        root=dataset_root,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name,
        mode=data_mode,
        enable_caa=False,
        return_original_mask=compute_original_size_metrics,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    results = {
        obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}
        for obj in eval_data.obj_list
    }
    original_size_results = None
    if compute_original_size_metrics:
        original_size_results = {
            obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}
            for obj in eval_data.obj_list
        }

    for batch in tqdm(eval_loader, desc=desc):
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
        if compute_original_size_metrics:
            orig_gt_mask = (batch["orig_img_mask"] > 0.5).float()
            orig_anomaly_map = F.interpolate(
                outputs["anomaly_map"].unsqueeze(1),
                size=orig_gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).cpu()
            original_size_results[cls_name]["gt_sp"].append(anomaly_label)
            original_size_results[cls_name]["pr_sp"].append(float(image_score.item()))
            original_size_results[cls_name]["imgs_masks"].append(orig_gt_mask)
            original_size_results[cls_name]["anomaly_maps"].append(orig_anomaly_map)

    if logger is not None:
        logger.info("Resized-mask evaluation (paper-aligned fixed-size masks):")
    resized_summary = compute_metrics(results, eval_data.obj_list, logger)

    if compute_original_size_metrics:
        if logger is not None:
            logger.info("Original-size evaluation (upsampled anomaly map vs raw mask):")
        original_size_summary = compute_metrics(original_size_results, eval_data.obj_list, logger)
        resized_summary["original_size"] = original_size_summary

    return resized_summary
