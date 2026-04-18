import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

from dataset import generate_class_info


class UniADetDataset(data.Dataset):
    def __init__(
        self,
        root,
        transform,
        target_transform,
        dataset_name,
        mode="test",
        enable_caa=False,
        caa_prob=0.5,
        caa_grid_sizes=(2, 3),
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.enable_caa = enable_caa
        self.caa_prob = caa_prob
        self.caa_grid_sizes = tuple(int(size) for size in caa_grid_sizes)

        meta_info = json.load(open(f"{self.root}/meta.json", "r"))
        meta_info = meta_info[mode]

        self.data_all = []
        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        self.class_to_indices = self._build_class_to_indices(self.data_all)

    def _build_class_to_indices(self, data_all: List[Dict]) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for index, item in enumerate(data_all):
            mapping.setdefault(item["cls_name"], []).append(index)
        return mapping

    def __len__(self):
        return self.length

    def _load_raw_item(self, index: int):
        data_item = self.data_all[index]
        img_path = os.path.join(self.root, data_item["img_path"])
        mask_path = os.path.join(self.root, data_item["mask_path"])

        image = Image.open(img_path).convert("RGB")
        if data_item["anomaly"] == 0:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8), mode="L")
        else:
            if os.path.isdir(mask_path):
                mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8), mode="L")
            else:
                mask_np = np.array(Image.open(mask_path).convert("L")) > 0
                mask = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")

        image = self.transform(image) if self.transform is not None else image
        mask = self.target_transform(mask) if self.target_transform is not None else mask
        mask = torch.zeros(1, image.shape[-2], image.shape[-1]) if mask is None else mask
        mask = (mask > 0.5).float()
        return image, mask, data_item

    def _grid_boundaries(self, size: int, grid_size: int) -> List[Tuple[int, int]]:
        boundaries = torch.linspace(0, size, steps=grid_size + 1).round().to(torch.int64).tolist()
        return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(grid_size)]

    def _resize_image(self, image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _resize_mask(self, mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(mask.unsqueeze(0), size=size, mode="nearest").squeeze(0)

    def _sample_same_class_indices(self, cls_name: str, anomaly: int, count: int) -> List[int]:
        candidate_indices = self.class_to_indices.get(cls_name, [])
        if anomaly == 0:
            candidate_indices = [idx for idx in candidate_indices if self.data_all[idx]["anomaly"] == 0]
        if not candidate_indices:
            candidate_indices = self.class_to_indices.get(cls_name, [])
        if not candidate_indices:
            return []
        return [random.choice(candidate_indices) for _ in range(count)]

    def _apply_grid_mosaic(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        cls_name: str,
        anomaly: int,
    ):
        grid_size = random.choice(self.caa_grid_sizes)
        h, w = image.shape[-2:]
        row_ranges = self._grid_boundaries(h, grid_size)
        col_ranges = self._grid_boundaries(w, grid_size)

        sample_indices = self._sample_same_class_indices(cls_name, anomaly, grid_size * grid_size - 1)
        tiles = [(image, mask, anomaly)]
        for sample_index in sample_indices:
            sampled_image, sampled_mask, sampled_item = self._load_raw_item(sample_index)
            tiles.append((sampled_image, sampled_mask, sampled_item["anomaly"]))

        random.shuffle(tiles)

        mosaic_image = torch.zeros_like(image)
        mosaic_mask = torch.zeros_like(mask)
        mosaic_anomaly = 0
        tile_cursor = 0

        for row_start, row_end in row_ranges:
            for col_start, col_end in col_ranges:
                tile_image, tile_mask, tile_anomaly = tiles[min(tile_cursor, len(tiles) - 1)]
                target_size = (row_end - row_start, col_end - col_start)
                tile_image = self._resize_image(tile_image, target_size)
                tile_mask = self._resize_mask(tile_mask, target_size)
                mosaic_image[:, row_start:row_end, col_start:col_end] = tile_image
                mosaic_mask[:, row_start:row_end, col_start:col_end] = tile_mask
                mosaic_anomaly = max(mosaic_anomaly, int(tile_anomaly))
                tile_cursor += 1

        return mosaic_image, (mosaic_mask > 0.5).float(), mosaic_anomaly

    def _apply_grid_crop(self, image: torch.Tensor, mask: torch.Tensor, anomaly: int):
        grid_size = random.choice(self.caa_grid_sizes)
        h, w = image.shape[-2:]
        row_ranges = self._grid_boundaries(h, grid_size)
        col_ranges = self._grid_boundaries(w, grid_size)

        candidates = []
        for row_start, row_end in row_ranges:
            for col_start, col_end in col_ranges:
                patch_mask = mask[:, row_start:row_end, col_start:col_end]
                contains_anomaly = bool(patch_mask.sum().item() > 0)
                if anomaly == 0 or contains_anomaly:
                    candidates.append((row_start, row_end, col_start, col_end))

        if not candidates:
            candidates = [
                (row_start, row_end, col_start, col_end)
                for row_start, row_end in row_ranges
                for col_start, col_end in col_ranges
            ]

        row_start, row_end, col_start, col_end = random.choice(candidates)
        crop_image = image[:, row_start:row_end, col_start:col_end]
        crop_mask = mask[:, row_start:row_end, col_start:col_end]
        crop_image = self._resize_image(crop_image, (h, w))
        crop_mask = self._resize_mask(crop_mask, (h, w))
        crop_anomaly = int(crop_mask.sum().item() > 0)
        return crop_image, (crop_mask > 0.5).float(), crop_anomaly

    def __getitem__(self, index):
        image, image_mask, data_item = self._load_raw_item(index)
        anomaly = int(data_item["anomaly"])
        cls_name = data_item["cls_name"]

        if self.enable_caa and random.random() < self.caa_prob:
            if random.random() < 0.5:
                image, image_mask, anomaly = self._apply_grid_mosaic(image, image_mask, cls_name, anomaly)
            else:
                image, image_mask, anomaly = self._apply_grid_crop(image, image_mask, anomaly)

        cls_id = self.class_name_map_class_id.get(cls_name, 0)
        return {
            "img": image,
            "img_mask": image_mask,
            "cls_name": cls_name,
            "anomaly": anomaly,
            "img_path": os.path.join(self.root, data_item["img_path"]),
            "cls_id": cls_id,
        }
