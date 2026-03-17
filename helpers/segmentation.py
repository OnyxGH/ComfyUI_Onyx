from __future__ import annotations

from collections import namedtuple
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


def ensure_image_batch(image: torch.Tensor) -> list[torch.Tensor]:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image input must be a torch.Tensor.")
    if image.ndim == 3:
        return [image.detach().cpu()]
    if image.ndim == 4:
        return [image[index].detach().cpu() for index in range(image.shape[0])]
    raise ValueError("Image tensor must be rank 3 or 4 (batch, height, width, channels).")


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def clamp_bbox(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    x1 = int(np.clip(np.floor(float(box[0])), 0, max(0, width - 1)))
    y1 = int(np.clip(np.floor(float(box[1])), 0, max(0, height - 1)))
    x2 = int(np.clip(np.ceil(float(box[2])), x1 + 1, width))
    y2 = int(np.clip(np.ceil(float(box[3])), y1 + 1, height))
    return x1, y1, x2, y2


def make_crop_region(
    image_width: int,
    image_height: int,
    bbox: tuple[int, int, int, int],
    crop_factor: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    factor = max(1.0, float(crop_factor))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_w = width * factor / 2.0
    half_h = height * factor / 2.0
    crop_x1 = int(np.clip(np.floor(center_x - half_w), 0, max(0, image_width - 1)))
    crop_y1 = int(np.clip(np.floor(center_y - half_h), 0, max(0, image_height - 1)))
    crop_x2 = int(np.clip(np.ceil(center_x + half_w), crop_x1 + 1, image_width))
    crop_y2 = int(np.clip(np.ceil(center_y + half_h), crop_y1 + 1, image_height))
    return crop_x1, crop_y1, crop_x2, crop_y2


def build_bbox_mask(
    crop_region: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
    dilation: int,
) -> torch.Tensor:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    mask = torch.zeros((crop_h, crop_w), dtype=torch.float32)

    rel_x1 = max(0, bbox[0] - crop_x1)
    rel_y1 = max(0, bbox[1] - crop_y1)
    rel_x2 = min(crop_w, max(rel_x1 + 1, bbox[2] - crop_x1))
    rel_y2 = min(crop_h, max(rel_y1 + 1, bbox[3] - crop_y1))
    mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0
    return dilate_mask(mask, dilation) if dilation > 0 else mask


def dilate_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation <= 0:
        return mask
    kernel_size = dilation * 2 + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)
    expanded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation)
    return (expanded > 0).float().squeeze()


def empty_image_like(reference: Optional[torch.Tensor]) -> torch.Tensor:
    channels = int(reference.shape[-1]) if reference is not None and reference.ndim >= 3 else 3
    dtype = reference.dtype if reference is not None else torch.float32
    device = reference.device if reference is not None else torch.device("cpu")
    return torch.zeros((1, 64, 64, channels), dtype=dtype, device=device)


def segments_to_image_list(source_image: torch.Tensor, segments: Sequence[SEG]) -> list[torch.Tensor]:
    if not segments:
        return [empty_image_like(source_image)]

    images: list[torch.Tensor] = []
    for seg in segments:
        tensor = _segment_to_image_tensor(source_image, seg)
        if tensor is not None:
            images.append(tensor)
    return images or [empty_image_like(source_image)]


def _segment_to_image_tensor(image: torch.Tensor, seg: SEG) -> Optional[torch.Tensor]:
    tensor = _value_to_image_tensor(seg.cropped_image)
    if tensor is not None:
        return tensor
    if seg.crop_region is None:
        return None
    return _crop_from_source(image, seg.crop_region)


def _value_to_image_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
    else:
        return None

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        return None
    if tensor.numel() == 0:
        return None
    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    return torch.clamp(tensor, 0.0, 1.0)


def _crop_from_source(image: torch.Tensor, crop_region: tuple[int, int, int, int]) -> Optional[torch.Tensor]:
    x1, y1, x2, y2 = crop_region
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.numel() == 0:
        return None
    return crop.unsqueeze(0).clone()