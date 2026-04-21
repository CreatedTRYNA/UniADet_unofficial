import math
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class UniADetZeroShotLateFusion(nn.Module):
    """
    Late-fusion variant of UniADet zero-shot.

    Difference from the original implementation:
    1. Aggregate multi-layer anomaly maps first.
    2. Aggregate multi-layer classification scores separately.
    3. Apply the final image-score fusion only once.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_layers: List[int],
        image_size: int,
        temperature: float = 0.07,
        score_fusion_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_layers = list(feature_layers)
        self.image_size = image_size
        self.temperature = temperature
        self.score_fusion_weight = score_fusion_weight
        self.feature_dim = backbone.embed_dim

        self.cls_weights = nn.ParameterDict(
            {
                f"layer_{layer}": nn.Parameter(torch.empty(2, self.feature_dim))
                for layer in self.feature_layers
            }
        )
        self.seg_weights = nn.ParameterDict(
            {
                f"layer_{layer}": nn.Parameter(torch.empty(2, self.feature_dim))
                for layer in self.feature_layers
            }
        )
        self.reset_parameters()

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def reset_parameters(self):
        for weight in self.cls_weights.values():
            nn.init.normal_(weight, std=0.02)
        for weight in self.seg_weights.values():
            nn.init.normal_(weight, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def extra_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }

    def _cosine_logits(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, eps=1e-6)
        weights = F.normalize(weights, dim=-1, eps=1e-6)
        return torch.matmul(features, weights.t()) / self.temperature

    def forward(self, image: torch.Tensor):
        with torch.no_grad():
            backbone_output = self.backbone.encode_image(image, self.feature_layers)

        cls_logits_per_layer = []
        cls_scores_per_layer = []
        seg_probs_per_layer = []
        anomaly_maps_per_layer = []

        for layer, cls_token, patch_tokens in zip(
            self.feature_layers,
            backbone_output["cls_tokens"],
            backbone_output["patch_tokens"],
        ):
            layer_key = f"layer_{layer}"

            cls_logits = self._cosine_logits(cls_token, self.cls_weights[layer_key])
            cls_score = F.softmax(cls_logits, dim=-1)[:, 1]

            seg_logits = self._cosine_logits(patch_tokens, self.seg_weights[layer_key])
            side = int(math.sqrt(seg_logits.shape[1]))
            if side * side != seg_logits.shape[1]:
                raise ValueError(
                    f"Patch token count {seg_logits.shape[1]} from layer {layer} does not form a square grid."
                )

            seg_logits = seg_logits.permute(0, 2, 1).reshape(seg_logits.shape[0], 2, side, side)
            seg_logits = F.interpolate(
                seg_logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            seg_probs = F.softmax(seg_logits, dim=1)
            anomaly_map = seg_probs[:, 1, :, :]

            cls_logits_per_layer.append(cls_logits)
            cls_scores_per_layer.append(cls_score)
            seg_probs_per_layer.append(seg_probs)
            anomaly_maps_per_layer.append(anomaly_map)

        final_anomaly_map = torch.stack(anomaly_maps_per_layer, dim=0).mean(dim=0)
        final_cls_score = torch.stack(cls_scores_per_layer, dim=0).mean(dim=0)
        final_map_score = final_anomaly_map.amax(dim=(1, 2))
        final_image_score = (
            (1.0 - self.score_fusion_weight) * final_cls_score
            + self.score_fusion_weight * final_map_score
        )

        return {
            "cls_logits_per_layer": cls_logits_per_layer,
            "cls_scores_per_layer": cls_scores_per_layer,
            "seg_probs_per_layer": seg_probs_per_layer,
            "anomaly_maps_per_layer": anomaly_maps_per_layer,
            "anomaly_map": final_anomaly_map,
            "cls_score": final_cls_score,
            "map_score": final_map_score,
            "image_score": final_image_score,
        }
