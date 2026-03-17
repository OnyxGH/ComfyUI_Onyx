from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .segmentation import SEG, dilate_mask, make_crop_region


def extract_resolution(tensor: torch.Tensor | None) -> tuple[int, int]:
    if tensor is None:
        return (64, 64)
    if tensor.ndim == 4:
        return int(tensor.shape[1]), int(tensor.shape[2])
    if tensor.ndim == 3:
        return int(tensor.shape[0]), int(tensor.shape[1])
    return (64, 64)


def sanitize_segs(value: Any, fallback: tuple[tuple[int, int], list[SEG]]) -> tuple[tuple[int, int], list[SEG]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return fallback
    shape_raw, segs_raw = value
    if not isinstance(segs_raw, (list, tuple)):
        return fallback
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 2:
        return fallback
    try:
        shape = (int(shape_raw[0]), int(shape_raw[1]))
    except Exception:
        return fallback
    return shape, list(segs_raw)


def build_seg(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, confidence: float, label: str | None, crop_factor: float, dilation: int) -> SEG | None:
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)
    if dilation > 0:
        mask_tensor = dilate_mask(mask_tensor.float(), dilation)
    active = mask_tensor > 0.5
    if not bool(active.any()):
        return None
    nonzero = torch.nonzero(active, as_tuple=False)
    y1, y2 = int(nonzero[:, 0].min().item()), int(nonzero[:, 0].max().item()) + 1
    x1, x2 = int(nonzero[:, 1].min().item()), int(nonzero[:, 1].max().item()) + 1
    bbox = (x1, y1, x2, y2)
    crop_region = make_crop_region(int(image_tensor.shape[1]), int(image_tensor.shape[0]), bbox, crop_factor)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    cropped_image = image_tensor[crop_y1:crop_y2, crop_x1:crop_x2].clone()
    if cropped_image.numel() == 0:
        return None
    cropped_mask = mask_tensor[crop_y1:crop_y2, crop_x1:crop_x2].detach().cpu().numpy().astype(np.float32)
    if cropped_mask.size == 0:
        return None
    return SEG(cropped_image, cropped_mask, float(confidence), crop_region, bbox, label)


def batchify_segments_for_sample(samples: list[torch.Tensor], segments: list[SEG], sample_index: int) -> list[SEG]:
    if not segments:
        return []
    batch_size = len(samples)
    sample = samples[sample_index]
    channels = int(sample.shape[-1]) if sample.ndim == 3 else 3
    batched_segments: list[SEG] = []
    for seg in segments:
        crop_x1, crop_y1, crop_x2, crop_y2 = seg.crop_region
        crop_h = max(0, crop_y2 - crop_y1)
        crop_w = max(0, crop_x2 - crop_x1)
        if crop_h <= 0 or crop_w <= 0:
            continue

        cropped_image = torch.zeros((batch_size, crop_h, crop_w, channels), dtype=sample.dtype, device=sample.device)
        source_crop = sample[crop_y1:crop_y2, crop_x1:crop_x2]
        if source_crop.shape[0] == crop_h and source_crop.shape[1] == crop_w:
            cropped_image[sample_index] = source_crop

        mask_np = np.asarray(seg.cropped_mask, dtype=np.float32)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if mask_np.ndim != 2 or mask_np.shape != (crop_h, crop_w):
            continue

        cropped_mask = np.zeros((batch_size, crop_h, crop_w), dtype=np.float32)
        cropped_mask[sample_index] = mask_np
        batched_segments.append(SEG(cropped_image, cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper))
    return batched_segments