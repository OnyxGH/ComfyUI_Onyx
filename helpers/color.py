from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..lib.color_correction import (
    adain_color_fix,
    lab_affine_color_correct_tensor,
    moment,
    paired_pixel_color_correction,
    root_poly_color_correct_tensor,
    wavelet_color_fix,
)
from .image import pil2tensor, tensor2pil

MATCH_COLOR_METHODS = [
    "wavelet",
    "adain",
    "moment",
    "poly",
    "ccm",
    "lab",
    "mkl",
    "hm",
    "reinhard",
    "mvgd",
    "hm-mvgd-hm",
    "hm-mkl-hm",
]
_COLOR_MATCHER_METHODS = {"mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"}


def resolve_reference_image(image_ref: torch.Tensor, batch_size: int, index: int) -> torch.Tensor:
    if image_ref.size(0) == 1:
        return image_ref[:1]
    if image_ref.size(0) != batch_size:
        raise ValueError("Match Color expects either a single reference image or a batch matching the target image batch size.")
    return image_ref[index : index + 1]


def resize_like(image: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[1:3]) == spatial_shape:
        return image

    channels_first = image.permute(0, 3, 1, 2)
    resized = F.interpolate(channels_first, size=spatial_shape, mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def match_with_color_matcher(image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> torch.Tensor:
    try:
        from color_matcher import ColorMatcher  # type: ignore
    except ImportError as exc:
        raise ImportError("Match Color requires the 'color-matcher' package for this method.") from exc

    matcher = ColorMatcher()
    outputs: list[torch.Tensor] = []
    batch_size = image_target.size(0)

    for index in range(batch_size):
        target_image = image_target[index : index + 1]
        reference_image = resolve_reference_image(image_ref, batch_size, index)
        if reference_image.shape[1:3] != target_image.shape[1:3]:
            reference_image = resize_like(reference_image, tuple(target_image.shape[1:3]))

        result = matcher.transfer(
            src=target_image[0].detach().cpu().numpy(),
            ref=reference_image[0].detach().cpu().numpy(),
            method=method,
        )
        outputs.append(torch.from_numpy(np.asarray(result, dtype=np.float32)))

    return torch.stack(outputs, dim=0)


def match_color(image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> torch.Tensor:
    batch_size = image_target.size(0)
    outputs: list[torch.Tensor] = []

    if method in {"wavelet", "adain"}:
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = resolve_reference_image(image_ref, batch_size, index)
            result_pil = (
                wavelet_color_fix(tensor2pil(target_image), tensor2pil(reference_image))
                if method == "wavelet"
                else adain_color_fix(tensor2pil(target_image), tensor2pil(reference_image))
            )
            outputs.append(pil2tensor(result_pil).to(image_target.device))
        return torch.cat(outputs, dim=0)

    if method == "moment":
        for index in range(batch_size):
            outputs.append(
                moment(
                    image=image_target[index : index + 1],
                    reference=resolve_reference_image(image_ref, batch_size, index),
                    strength=1.0,
                    adaptive_matching=True,
                    device=image_target.device,
                )
            )
        return torch.cat(outputs, dim=0)

    if method == "ccm":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = resolve_reference_image(image_ref, batch_size, index)
            outputs.append(paired_pixel_color_correction(target_image, reference_image, degree=1))
        return torch.cat(outputs, dim=0)

    if method == "poly":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = resolve_reference_image(image_ref, batch_size, index)
            if reference_image.shape[1:3] != target_image.shape[1:3]:
                reference_image = resize_like(reference_image, tuple(target_image.shape[1:3]))
            outputs.append(root_poly_color_correct_tensor(reference_image, target_image, 0.1, 200_000, 1e-3))
        return torch.cat(outputs, dim=0)

    if method == "lab":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = resolve_reference_image(image_ref, batch_size, index)
            if reference_image.shape[1:3] != target_image.shape[1:3]:
                reference_image = resize_like(reference_image, tuple(target_image.shape[1:3]))
            outputs.append(lab_affine_color_correct_tensor(reference_image, target_image, 0.1, 200_000, 1e-3))
        return torch.cat(outputs, dim=0)

    if method in _COLOR_MATCHER_METHODS:
        return match_with_color_matcher(image_ref, image_target, method).to(image_target.device)

    raise ValueError(f"Unsupported Match Color method: {method}")