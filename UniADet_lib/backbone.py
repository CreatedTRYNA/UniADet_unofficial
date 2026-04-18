import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from VisualAD_lib.VisualAD import LayerNorm, Transformer
from VisualAD_lib import model_load as clip_model_load
from VisualAD_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .dinov2_fallback import load_dinov2_reg_vitl14_backbone


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str
    model_name: str
    default_image_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    patch_size: int | None = None
    alias_of: str | None = None
    note: str | None = None


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "ViT-B/32": BackboneSpec(
        name="ViT-B/32",
        kind="clip",
        model_name="ViT-B/32",
        default_image_size=518,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
        patch_size=32,
    ),
    "ViT-B/16": BackboneSpec(
        name="ViT-B/16",
        kind="clip",
        model_name="ViT-B/16",
        default_image_size=518,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
        patch_size=16,
    ),
    "ViT-L/14": BackboneSpec(
        name="ViT-L/14",
        kind="clip",
        model_name="ViT-L/14",
        default_image_size=518,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
        patch_size=14,
    ),
    "ViT-L/14@336px": BackboneSpec(
        name="ViT-L/14@336px",
        kind="clip",
        model_name="ViT-L/14@336px",
        default_image_size=518,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
        patch_size=14,
    ),
    "DINOv2-R ViT-L/14": BackboneSpec(
        name="DINOv2-R ViT-L/14",
        kind="timm",
        model_name="vit_large_patch14_reg4_dinov2",
        default_image_size=518,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        patch_size=14,
    ),
    "vit_large_patch14_reg4_dinov2": BackboneSpec(
        name="vit_large_patch14_reg4_dinov2",
        kind="timm",
        model_name="vit_large_patch14_reg4_dinov2",
        default_image_size=518,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        patch_size=14,
        alias_of="DINOv2-R ViT-L/14",
    ),
    "DINOv2-R ViT-L/16": BackboneSpec(
        name="DINOv2-R ViT-L/16",
        kind="timm",
        model_name="vit_large_patch14_reg4_dinov2",
        default_image_size=518,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        patch_size=14,
        alias_of="DINOv2-R ViT-L/14",
        note="The UniADet supplementary uses DINOv2-R ViT-L/14; this alias maps to that backbone.",
    ),
    "DINOv3 ViT-L/16": BackboneSpec(
        name="DINOv3 ViT-L/16",
        kind="timm",
        model_name="vit_large_patch16_dinov3",
        default_image_size=512,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        patch_size=16,
    ),
    "vit_large_patch16_dinov3": BackboneSpec(
        name="vit_large_patch16_dinov3",
        kind="timm",
        model_name="vit_large_patch16_dinov3",
        default_image_size=512,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        patch_size=16,
        alias_of="DINOv3 ViT-L/16",
    ),
}


def resolve_backbone_spec(name: str) -> BackboneSpec:
    if name not in _BACKBONE_SPECS:
        raise RuntimeError(f"Unsupported backbone {name}; available models = {available_models()}")
    spec = _BACKBONE_SPECS[name]
    if spec.note:
        warnings.warn(spec.note)
    return spec


def get_backbone_data_config(name: str, image_size: int | None = None) -> Dict[str, Union[str, int, Tuple[float, ...]]]:
    spec = resolve_backbone_spec(name)
    resolved_image_size = image_size or spec.default_image_size
    return {
        "name": spec.alias_of or spec.name,
        "kind": spec.kind,
        "model_name": spec.model_name,
        "image_size": resolved_image_size,
        "mean": spec.mean,
        "std": spec.std,
        "patch_size": spec.patch_size,
    }


class CLIPVisionBackbone(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.embed_dim = width
        self.total_layers = layers

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        num_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, width // 64)
        self.ln_post = LayerNorm(width)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def _interpolate_positional_embedding(self, token_count: int) -> torch.Tensor:
        base_side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((token_count - 1) ** 0.5)

        if base_side == new_side:
            return self.positional_embedding.unsqueeze(0)

        class_pos = self.positional_embedding[:1, :].unsqueeze(0)
        patch_pos = self.positional_embedding[1:, :].reshape(1, base_side, base_side, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_side, new_side),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos = patch_pos.reshape(1, self.embed_dim, new_side * new_side).permute(0, 2, 1)
        return torch.cat([class_pos, patch_pos], dim=1)

    def encode_image(self, image: torch.Tensor, feature_list: List[int]):
        x = self.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        class_token = self.class_embedding.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(1).to(x.dtype)
        x = torch.cat([class_token, x], dim=1)
        x = x + self._interpolate_positional_embedding(x.shape[1]).to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        _, out_tokens = self.transformer.forward_dispatch(x, feature_list)

        cls_tokens = []
        patch_tokens = []
        for tokens in out_tokens:
            tokens = self.ln_post(tokens.permute(1, 0, 2))
            cls_tokens.append(tokens[:, 0, :])
            patch_tokens.append(tokens[:, 1:, :])

        return {
            "cls_tokens": cls_tokens,
            "patch_tokens": patch_tokens,
            "patch_start_idx": 1,
        }


class TimmVisionBackbone(nn.Module):
    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.model = timm_model
        self.embed_dim = getattr(timm_model, "embed_dim", getattr(timm_model, "num_features"))
        self.total_layers = len(timm_model.blocks)

        patch_size = getattr(timm_model.patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = patch_size

        img_size = getattr(timm_model.patch_embed, "img_size", None)
        if isinstance(img_size, tuple):
            img_size = img_size[0]
        self.input_resolution = img_size

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def encode_image(self, image: torch.Tensor, feature_list: List[int]):
        indices = [layer - 1 for layer in feature_list]
        final_tokens, intermediates = self.model.forward_intermediates(
            image.type(self.dtype),
            indices=indices,
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
        )

        cls_tokens = []
        patch_tokens = []
        for patch_feature, prefix_tokens in intermediates:
            cls_tokens.append(prefix_tokens[:, 0, :])
            patch_tokens.append(patch_feature)

        return {
            "cls_tokens": cls_tokens,
            "patch_tokens": patch_tokens,
            "patch_start_idx": getattr(self.model, "num_prefix_tokens", 1),
            "final_tokens": final_tokens,
        }


def _load_clip_state_dict(
    name: str,
    download_root: Union[str, None] = None,
) -> dict:
    if name in clip_model_load._MODELS:
        model_path = clip_model_load._download(
            clip_model_load._MODELS[name],
            download_root or os.path.expanduser("~/.cache/clip"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, "rb") as opened_file:
        try:
            jit_model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            opened_file.seek(0)
            state_dict = torch.load(opened_file, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def available_models() -> List[str]:
    return list(_BACKBONE_SPECS.keys())


def _freeze_model(model: nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_clip_backbone(
    name: str,
    device: Union[str, torch.device] = "cpu",
    download_root: Union[str, None] = None,
):
    state_dict = _load_clip_state_dict(name, download_root=download_root)

    if "visual.proj" not in state_dict:
        raise RuntimeError("Only CLIP ViT backbones are supported by the current UniADet implementation.")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            key
            for key in state_dict.keys()
            if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    model = CLIPVisionBackbone(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
    )

    vision_state_dict = {
        key[len("visual."):]: value
        for key, value in state_dict.items()
        if key.startswith("visual.")
    }
    vision_state_dict.pop("proj", None)

    incompatible = model.load_state_dict(vision_state_dict, strict=False)
    if incompatible.unexpected_keys:
        warnings.warn(f"Unexpected CLIP vision keys ignored: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        warnings.warn(f"Missing CLIP vision keys during load: {incompatible.missing_keys}")

    model.to(device)
    return _freeze_model(model)


def load_timm_backbone(
    name: str,
    device: Union[str, torch.device] = "cpu",
    image_size: int | None = None,
    download_root: Union[str, None] = None,
    pretrained: bool = True,
):
    spec = resolve_backbone_spec(name)
    config = get_backbone_data_config(name, image_size=image_size)

    # Use the repository-local DINOv2 loader for the register ViT-L/14 backbone.
    # This avoids flaky HF downloads inside timm and keeps behavior stable across
    # timm versions while still matching timm numerically.
    if spec.model_name == "vit_large_patch14_reg4_dinov2":
        model = load_dinov2_reg_vitl14_backbone(
            device=device,
            image_size=config["image_size"],
            pretrained=pretrained,
            download_root=download_root,
        )
        return _freeze_model(model)

    import timm
    timm_version = getattr(timm, "__version__", "unknown")

    try:
        timm_model = timm.create_model(
            spec.model_name,
            pretrained=pretrained,
            img_size=config["image_size"],
        )
    except Exception as exc:
        if spec.model_name == "vit_large_patch14_reg4_dinov2":
            warnings.warn(
                f"timm {timm_version} does not register {spec.model_name}; "
                "falling back to the repository-local DINOv2 loader."
            )
            model = load_dinov2_reg_vitl14_backbone(
                device=device,
                image_size=config["image_size"],
                pretrained=pretrained,
                download_root=download_root,
            )
            return _freeze_model(model)

        raise RuntimeError(
            f"Failed to load pretrained backbone {spec.model_name} with timm {timm_version}. "
            "If this is the first run, the pretrained weights may need to be downloaded. "
            "DINOv2/DINOv3 backbones require timm>=1.0.0 unless a repository-local fallback is available."
        ) from exc

    if not hasattr(timm_model, "forward_intermediates"):
        if spec.model_name == "vit_large_patch14_reg4_dinov2":
            warnings.warn(
                f"timm {timm_version} loaded {spec.model_name} but does not provide forward_intermediates; "
                "falling back to the repository-local DINOv2 loader."
            )
            model = load_dinov2_reg_vitl14_backbone(
                device=device,
                image_size=config["image_size"],
                pretrained=pretrained,
                download_root=download_root,
            )
            return _freeze_model(model)

        raise RuntimeError(
            f"Backbone {spec.model_name} loaded with timm {timm_version}, but this timm build "
            "does not expose forward_intermediates, which UniADet needs for multi-layer features. "
            "Please upgrade timm to >=1.0.0."
        )

    model = TimmVisionBackbone(timm_model).to(device)
    return _freeze_model(model)


def load_backbone(
    name: str,
    device: Union[str, torch.device] = "cpu",
    image_size: int | None = None,
    download_root: Union[str, None] = None,
    pretrained: bool = True,
):
    spec = resolve_backbone_spec(name)
    if spec.kind == "clip":
        return load_clip_backbone(spec.model_name, device=device, download_root=download_root)
    if spec.kind == "timm":
        return load_timm_backbone(
            name,
            device=device,
            image_size=image_size,
            download_root=download_root,
            pretrained=pretrained,
        )
    raise RuntimeError(f"Unsupported backbone kind: {spec.kind}")
