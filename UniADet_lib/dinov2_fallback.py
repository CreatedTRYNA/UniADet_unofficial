import math
import warnings
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


_DINOV2_VITL14_REG4_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        grid_size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, init_values: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2RegVisionBackbone(nn.Module):
    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_reg_tokens: int = 4,
        init_values: float = 1e-5,
    ):
        super().__init__()
        self.input_resolution = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.total_layers = depth
        self.num_prefix_tokens = 1 + num_reg_tokens

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads=num_heads, mlp_ratio=4.0, init_values=init_values) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.reg_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _interpolate_pos_embed(self, grid_size: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        target_h, target_w = grid_size
        target_tokens = target_h * target_w
        if self.pos_embed.shape[1] == target_tokens:
            return self.pos_embed.to(dtype=dtype)

        source_tokens = self.pos_embed.shape[1]
        source_side = int(math.sqrt(source_tokens))
        if source_side * source_side != source_tokens:
            raise RuntimeError(f"Unexpected DINOv2 positional embedding size: {self.pos_embed.shape}")

        pos_embed = self.pos_embed.reshape(1, source_side, source_side, self.embed_dim).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed.float(),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, target_tokens, self.embed_dim)
        return pos_embed.to(dtype=dtype)

    def _pos_embed(self, patch_tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        pos_embed = self._interpolate_pos_embed(grid_size, patch_tokens.dtype)
        patch_tokens = patch_tokens + pos_embed

        prefix_tokens = [
            self.cls_token.expand(patch_tokens.shape[0], -1, -1),
            self.reg_token.expand(patch_tokens.shape[0], -1, -1),
        ]
        return torch.cat(prefix_tokens + [patch_tokens], dim=1)

    def encode_image(self, image: torch.Tensor, feature_list: List[int]) -> Dict[str, Union[List[torch.Tensor], int]]:
        indices = [layer - 1 for layer in feature_list]
        requested = set(indices)

        patch_tokens, grid_size = self.patch_embed(image.to(dtype=self.dtype))
        x = self._pos_embed(patch_tokens, grid_size)

        intermediates = {}
        for block_index, block in enumerate(self.blocks):
            x = block(x)
            if block_index in requested:
                intermediates[block_index] = self.norm(x)

        final_tokens = self.norm(x)

        cls_tokens = []
        patch_tokens = []
        for block_index in indices:
            tokens = intermediates[block_index]
            cls_tokens.append(tokens[:, 0, :])
            patch_tokens.append(tokens[:, self.num_prefix_tokens:, :])

        return {
            "cls_tokens": cls_tokens,
            "patch_tokens": patch_tokens,
            "patch_start_idx": self.num_prefix_tokens,
            "final_tokens": final_tokens,
        }


def _unwrap_state_dict(state_dict: dict) -> dict:
    for container_key in ("state_dict", "model", "teacher"):
        if container_key in state_dict and isinstance(state_dict[container_key], dict):
            state_dict = state_dict[container_key]

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    if state_dict and all(key.startswith("backbone.") for key in state_dict):
        state_dict = {key[len("backbone."):]: value for key, value in state_dict.items()}
    return state_dict


def _adapt_official_dinov2_state_dict(state_dict: dict, model: DinoV2RegVisionBackbone) -> dict:
    state_dict = dict(state_dict)

    # Official DINOv2 checkpoints name register tokens differently from timm.
    if "register_tokens" in state_dict and "reg_token" not in state_dict:
        state_dict["reg_token"] = state_dict.pop("register_tokens")

    # Not used by this frozen vision-only backbone.
    state_dict.pop("mask_token", None)

    pos_embed = state_dict.get("pos_embed")
    if pos_embed is not None:
        expected_tokens = model.pos_embed.shape[1]

        # Official checkpoints include a class positional embedding, while timm's
        # DINOv2 reg models keep class/register token embeddings separate.
        if pos_embed.shape[1] == expected_tokens + 1:
            pos_embed = pos_embed[:, 1:, :]

        if pos_embed.shape[1] != expected_tokens:
            source_tokens = pos_embed.shape[1]
            source_side = int(math.sqrt(source_tokens))
            target_side = int(math.sqrt(expected_tokens))
            if source_side * source_side != source_tokens or target_side * target_side != expected_tokens:
                raise RuntimeError(
                    f"Unexpected DINOv2 positional embedding size: checkpoint={tuple(pos_embed.shape)}, "
                    f"model={tuple(model.pos_embed.shape)}"
                )

            pos_embed = pos_embed.reshape(1, source_side, source_side, model.embed_dim).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=(target_side, target_side),
                mode="bicubic",
                align_corners=False,
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, expected_tokens, model.embed_dim)

        state_dict["pos_embed"] = pos_embed.to(dtype=model.pos_embed.dtype)

    return state_dict


def load_dinov2_reg_vitl14_backbone(
    device: Union[str, torch.device] = "cpu",
    image_size: int | None = None,
    pretrained: bool = True,
    download_root: Union[str, None] = None,
):
    requested_image_size = image_size or 518
    model = DinoV2RegVisionBackbone(image_size=518 if pretrained else requested_image_size)

    if pretrained:
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                _DINOV2_VITL14_REG4_URL,
                map_location="cpu",
                model_dir=download_root,
                progress=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download/load pretrained DINOv2 ViT-L/14 register weights. "
                "Please check network access or place the checkpoint in the torch hub cache."
            ) from exc

        state_dict = _unwrap_state_dict(state_dict)
        state_dict = _adapt_official_dinov2_state_dict(state_dict, model)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            warnings.warn(f"Unexpected DINOv2 keys ignored: {incompatible.unexpected_keys}")
        if incompatible.missing_keys:
            warnings.warn(f"Missing DINOv2 keys during load: {incompatible.missing_keys}")

    # Keep metadata aligned with the requested input size; positional embeddings
    # are interpolated dynamically at runtime.
    model.input_resolution = requested_image_size
    model.patch_embed.img_size = (requested_image_size, requested_image_size)
    model.patch_embed.grid_size = (
        requested_image_size // model.patch_size,
        requested_image_size // model.patch_size,
    )

    return model.to(device)
