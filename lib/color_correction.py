from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from ..helpers.image import pil2tensor, tensor2pil

_CCM_CHUNK_SIZE = 262144
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ).astype(np.float32)
_X_N = 0.95047
_Y_N = 1.0
_Z_N = 1.08883
_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA**3
_INV_3DELTA2 = 1.0 / (3.0 * _DELTA * _DELTA)


def supports_amp() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


def calc_mean_std(feat: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    if feat.dim() != 4:
        raise ValueError("Expected a 4D tensor for mean/std calculation.")

    batch, channels = feat.shape[:2]
    feat_var = feat.view(batch, channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(batch, channels, 1, 1)
    feat_mean = feat.view(batch, channels, -1).mean(dim=2).view(batch, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    target_tensor = pil2tensor(target).permute(0, 3, 1, 2)
    source_tensor = pil2tensor(source).permute(0, 3, 1, 2)
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    return tensor2pil(result_tensor.permute(0, 2, 3, 1).clamp_(0.0, 1.0))


def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    kernel = torch.tensor(
        [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ],
        dtype=image.dtype,
        device=image.device,
    )[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def wavelet_decomposition(image: Tensor, levels: int = 5) -> tuple[Tensor, Tensor]:
    high_freq = torch.zeros_like(image)
    low_freq = image
    for level in range(levels):
        radius = 2**level
        low_freq = wavelet_blur(low_freq, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    target_tensor = pil2tensor(target).permute(0, 3, 1, 2)
    source_tensor = pil2tensor(source).permute(0, 3, 1, 2)
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    return tensor2pil(result_tensor.permute(0, 2, 3, 1).clamp_(0.0, 1.0))


def paired_pixel_color_correction(
    target: torch.Tensor,
    reference: torch.Tensor,
    degree: int = 1,
    sample_pixels: int = 120000,
    regularization: float = 1e-4,
    chunk_size: int = _CCM_CHUNK_SIZE,
) -> torch.Tensor:
    if degree < 1:
        raise ValueError("degree must be >= 1")

    target_device = target.device
    target_cpu = target.detach().to(torch.float32).cpu()
    reference_cpu = reference.detach().to(torch.float32).cpu()

    target_img = _prepare_ccm_image(target_cpu)
    reference_img = _prepare_ccm_image(reference_cpu)
    reference_img = _match_reference_resolution(reference_img, target_img.shape[:2])

    weights = _fit_polynomial_ccm(target_img, reference_img, degree, sample_pixels, regularization)
    corrected = _apply_polynomial_ccm(target_img, weights, degree, chunk_size)

    if target.dim() == 4:
        corrected = corrected.unsqueeze(0)

    return corrected.clamp_(0.0, 1.0).to(target_device)


def _prepare_ccm_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        if image.size(0) != 1:
            raise ValueError("paired_pixel_color_correction expects a single image tensor per call")
        image = image.squeeze(0)

    if image.dim() != 3:
        raise ValueError("paired_pixel_color_correction expects 3D image tensors")

    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)

    if image.shape[-1] != 3:
        raise ValueError("Image tensors must have 3 color channels")

    return image.contiguous()


def _match_reference_resolution(reference: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    if reference.shape[:2] == tuple(spatial_shape):
        return reference

    ref = reference.permute(2, 0, 1).unsqueeze(0)
    ref = F.interpolate(ref, size=spatial_shape, mode="bilinear", align_corners=False)
    return ref.squeeze(0).permute(1, 2, 0)


def _fit_polynomial_ccm(
    target: torch.Tensor,
    reference: torch.Tensor,
    degree: int,
    sample_pixels: int,
    regularization: float,
) -> torch.Tensor:
    src = target.view(-1, 3)
    dst = reference.view(-1, 3)
    num_pixels = src.size(0)

    if sample_pixels and num_pixels > sample_pixels:
        indices = torch.randperm(num_pixels, device=src.device)[:sample_pixels]
        src = src[indices]
        dst = dst[indices]

    features = _build_polynomial_features(src, degree)
    xtx = features.transpose(0, 1) @ features
    if regularization > 0:
        xtx = xtx + regularization * torch.eye(xtx.size(0), dtype=xtx.dtype, device=xtx.device)
    xty = features.transpose(0, 1) @ dst

    try:
        return torch.linalg.solve(xtx, xty)
    except RuntimeError:
        return torch.linalg.pinv(xtx) @ xty


def _apply_polynomial_ccm(target: torch.Tensor, weights: torch.Tensor, degree: int, chunk_size: int) -> torch.Tensor:
    flat = target.view(-1, 3)
    outputs: list[torch.Tensor] = []
    step = max(1, chunk_size or flat.size(0))

    for offset in range(0, flat.size(0), step):
        features = _build_polynomial_features(flat[offset : offset + step], degree)
        outputs.append(features @ weights)

    return torch.cat(outputs, dim=0).view_as(target)


@lru_cache(maxsize=None)
def _monomial_exponents(degree: int) -> tuple[tuple[int, int, int], ...]:
    exponents: list[tuple[int, int, int]] = [(0, 0, 0)]
    for current_degree in range(1, degree + 1):
        for combo in combinations_with_replacement(range(3), current_degree):
            counts = [0, 0, 0]
            for idx in combo:
                counts[idx] += 1
            exponents.append((counts[0], counts[1], counts[2]))
    return tuple(exponents)


def _build_polynomial_features(rgb: torch.Tensor, degree: int) -> torch.Tensor:
    features: list[torch.Tensor] = []
    ones = torch.ones(rgb.size(0), dtype=rgb.dtype, device=rgb.device)

    for exponents in _monomial_exponents(degree):
        if sum(exponents) == 0:
            features.append(ones)
            continue

        term = ones
        for channel, power in enumerate(exponents):
            if power:
                term = term * rgb[:, channel].pow(power)
        features.append(term)

    return torch.stack(features, dim=1)


def _ensure_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    if arr.max() > 1.5:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def _build_root_poly_features(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    rg = np.sqrt(np.clip(r * g, 0.0, 1.0))
    gb = np.sqrt(np.clip(g * b, 0.0, 1.0))
    rb = np.sqrt(np.clip(r * b, 0.0, 1.0))
    ones = np.ones_like(r)
    return np.stack([r, g, b, rg, gb, rb, ones], axis=1)


def fit_root_poly_color_mapping(
    ref_img: np.ndarray,
    proc_img: np.ndarray,
    sample_fraction: float = 0.1,
    max_samples: int = 200_000,
    ridge_lambda: float = 1e-3,
) -> dict[str, np.ndarray | float]:
    ref = _ensure_float01(ref_img)
    proc = _ensure_float01(proc_img)

    if ref.shape != proc.shape:
        raise ValueError(f"ref_img and proc_img must have same shape; got {ref.shape} vs {proc.shape}")
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError("Images must have shape (H, W, 3).")

    num_pixels = int(np.prod(ref.shape[:2]))
    ref_flat = ref.reshape(-1, 3)
    proc_flat = proc.reshape(-1, 3)

    if sample_fraction < 1.0 or num_pixels > max_samples:
        sample_count = min(int(num_pixels * sample_fraction), int(max_samples))
        sample_count = max(1, min(num_pixels, sample_count if sample_count >= 10 else int(max_samples)))
        indices = np.random.choice(num_pixels, size=sample_count, replace=False)
        ref_sample = ref_flat[indices]
        proc_sample = proc_flat[indices]
    else:
        ref_sample = ref_flat
        proc_sample = proc_flat

    features = _build_root_poly_features(proc_sample)
    targets = ref_sample
    xtx = features.T @ features
    xtx += ridge_lambda * np.eye(xtx.shape[0], dtype=np.float32)
    xty = features.T @ targets
    weights = np.linalg.solve(xtx, xty).astype(np.float32)
    return {"W": weights, "ridge_lambda": float(ridge_lambda)}


def apply_root_poly_color_mapping(proc_img: np.ndarray, mapping: dict[str, np.ndarray | float]) -> np.ndarray:
    proc = _ensure_float01(proc_img)
    height, width = proc.shape[:2]
    features = _build_root_poly_features(proc.reshape(-1, 3))
    weights = mapping["W"]
    corrected = features @ weights
    return np.clip(corrected, 0.0, 1.0).reshape(height, width, 3).astype(np.float32)


def root_poly_color_correct_tensor(
    image_ref: torch.Tensor,
    image_target: torch.Tensor,
    sample_fraction: float = 0.1,
    max_samples: int = 200_000,
    ridge_lambda: float = 1e-3,
) -> torch.Tensor:
    if image_ref.ndim != 4 or image_ref.shape[-1] != 3:
        raise ValueError(f"image_ref must be (B,H,W,3), got {tuple(image_ref.shape)}")
    if image_target.ndim != 4 or image_target.shape[-1] != 3:
        raise ValueError(f"image_target must be (B,H,W,3), got {tuple(image_target.shape)}")
    if image_ref.shape != image_target.shape:
        raise ValueError(f"image_ref and image_target shapes must match; got {tuple(image_ref.shape)} vs {tuple(image_target.shape)}")

    ref_np = image_ref[0].detach().cpu().numpy()
    target_np = image_target[0].detach().cpu().numpy()
    mapping = fit_root_poly_color_mapping(ref_np, target_np, sample_fraction, max_samples, ridge_lambda)
    corrected = apply_root_poly_color_mapping(target_np, mapping)
    return torch.from_numpy(corrected).unsqueeze(0).to(image_target.device)


def _srgb_to_linear_np(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.empty_like(rgb)
    low = rgb <= 0.04045
    out[low] = rgb[low] / 12.92
    out[~low] = ((rgb[~low] + 0.055) / 1.055) ** 2.4
    return out


def _linear_to_srgb_np(rgb_lin: np.ndarray) -> np.ndarray:
    rgb_lin = np.asarray(rgb_lin, dtype=np.float32)
    out = np.empty_like(rgb_lin)
    low = rgb_lin <= 0.0031308
    out[low] = 12.92 * rgb_lin[low]
    out[~low] = 1.055 * (rgb_lin[~low] ** (1.0 / 2.4)) - 0.055
    return out


def _xyz_to_lab_np(xyz: np.ndarray) -> np.ndarray:
    x = xyz[..., 0] / _X_N
    y = xyz[..., 1] / _Y_N
    z = xyz[..., 2] / _Z_N

    def f(value: np.ndarray) -> np.ndarray:
        out = np.empty_like(value)
        high = value > _DELTA_CUBED
        out[high] = np.cbrt(value[high])
        out[~high] = value[~high] * _INV_3DELTA2 + 4.0 / 29.0
        return out

    fx = f(x)
    fy = f(y)
    fz = f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _lab_to_xyz_np(lab: np.ndarray) -> np.ndarray:
    l_value = lab[..., 0]
    a_value = lab[..., 1]
    b_value = lab[..., 2]

    fy = (l_value + 16.0) / 116.0
    fx = fy + a_value / 500.0
    fz = fy - b_value / 200.0

    def finv(value: np.ndarray) -> np.ndarray:
        out = np.empty_like(value)
        high = value > _DELTA
        out[high] = value[high] ** 3
        out[~high] = 3.0 * _DELTA * _DELTA * (value[~high] - 4.0 / 29.0)
        return out

    return np.stack([_X_N * finv(fx), _Y_N * finv(fy), _Z_N * finv(fz)], axis=-1)


def _rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    rgb = _ensure_float01(rgb)
    rgb_lin = _srgb_to_linear_np(rgb)
    xyz = rgb_lin.reshape(-1, 3) @ _RGB_TO_XYZ.T
    return _xyz_to_lab_np(xyz).reshape(rgb.shape)


def _lab_to_rgb_np(lab: np.ndarray) -> np.ndarray:
    xyz = _lab_to_xyz_np(lab.reshape(-1, 3))
    rgb_lin = xyz @ _XYZ_TO_RGB.T
    return np.clip(_linear_to_srgb_np(rgb_lin).reshape(lab.shape), 0.0, 1.0)


def fit_lab_affine_mapping(
    ref_img: np.ndarray,
    proc_img: np.ndarray,
    sample_fraction: float = 0.1,
    max_samples: int = 200_000,
    ridge_lambda: float = 1e-3,
) -> dict[str, np.ndarray | float]:
    ref = _ensure_float01(ref_img)
    proc = _ensure_float01(proc_img)
    if ref.shape != proc.shape or ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError(f"ref_img and proc_img must both be (H,W,3); got {ref.shape} vs {proc.shape}")

    num_pixels = int(np.prod(ref.shape[:2]))
    ref_lab = _rgb_to_lab_np(ref).reshape(-1, 3)
    proc_lab = _rgb_to_lab_np(proc).reshape(-1, 3)

    if sample_fraction < 1.0 or num_pixels > max_samples:
        sample_count = min(int(num_pixels * sample_fraction), max_samples)
        sample_count = max(1, min(num_pixels, sample_count if sample_count >= 10 else max_samples))
        indices = np.random.choice(num_pixels, size=sample_count, replace=False)
        source = proc_lab[indices]
        target = ref_lab[indices]
    else:
        source = proc_lab
        target = ref_lab

    mu_src = source.mean(axis=0, dtype=np.float64)
    mu_dst = target.mean(axis=0, dtype=np.float64)
    centered_src = source - mu_src
    centered_dst = target - mu_dst
    xtx = centered_src.T @ centered_src + ridge_lambda * np.eye(3, dtype=np.float64)
    xty = centered_src.T @ centered_dst
    matrix = np.linalg.solve(xtx, xty).astype(np.float32)
    bias = (mu_dst - mu_src @ matrix).astype(np.float32)
    return {"A": matrix, "b": bias, "ridge_lambda": float(ridge_lambda)}


def apply_lab_affine_mapping(proc_img: np.ndarray, mapping: dict[str, np.ndarray | float]) -> np.ndarray:
    proc = _ensure_float01(proc_img)
    height, width = proc.shape[:2]
    lab = _rgb_to_lab_np(proc).reshape(-1, 3)
    corrected_lab = (lab @ mapping["A"] + mapping["b"]).reshape(height, width, 3)
    corrected_lab[..., 0] = np.clip(corrected_lab[..., 0], 0.0, 100.0)
    corrected_lab[..., 1:] = np.clip(corrected_lab[..., 1:], -128.0, 127.0)
    return _lab_to_rgb_np(corrected_lab).astype(np.float32)


def lab_affine_color_correct_tensor(
    image_ref: torch.Tensor,
    image_target: torch.Tensor,
    sample_fraction: float = 0.1,
    max_samples: int = 200_000,
    ridge_lambda: float = 1e-3,
) -> torch.Tensor:
    if image_ref.ndim != 4 or image_ref.shape[-1] != 3:
        raise ValueError(f"image_ref must be (B,H,W,3), got {tuple(image_ref.shape)}")
    if image_target.ndim != 4 or image_target.shape[-1] != 3:
        raise ValueError(f"image_target must be (B,H,W,3), got {tuple(image_target.shape)}")
    if image_ref.shape != image_target.shape:
        raise ValueError(f"image_ref and image_target must have matching shapes; got {tuple(image_ref.shape)} vs {tuple(image_target.shape)}")

    outputs: list[torch.Tensor] = []
    for batch_index in range(image_target.shape[0]):
        ref_np = image_ref[batch_index].detach().cpu().numpy()
        target_np = image_target[batch_index].detach().cpu().numpy()
        mapping = fit_lab_affine_mapping(ref_np, target_np, sample_fraction, max_samples, ridge_lambda)
        corrected = apply_lab_affine_mapping(target_np, mapping)
        outputs.append(torch.from_numpy(corrected).to(image_target.device).unsqueeze(0))

    return torch.cat(outputs, dim=0)


def match_channel(input_channel: torch.Tensor, ref_channel: torch.Tensor, strength: float, adaptive_enabled: bool) -> torch.Tensor:
    input_mean = input_channel.float().mean()
    input_std = input_channel.float().std()
    ref_mean = ref_channel.float().mean()
    ref_std = ref_channel.float().std()
    normalized = (input_channel - input_mean) / (input_std + 1e-8)
    matched_channel = normalized * ref_std + ref_mean

    if adaptive_enabled:
        mean_diff = torch.abs(input_mean - ref_mean)
        std_diff = torch.abs(input_std - ref_std)
        adjustment = (2.0 * mean_diff + std_diff) / 3.00001
        final_strength = torch.clamp(strength * (1 - adjustment), 0.0, 1.0)
    else:
        final_strength = torch.as_tensor(strength, device=input_channel.device, dtype=input_channel.dtype)

    return torch.lerp(input_channel, matched_channel, final_strength)


def moment(
    image: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
    adaptive_matching: bool,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if strength == 0:
        return image

    target_device = torch.device(device) if device is not None else image.device
    use_amp = supports_amp() and target_device.type == "cuda"
    working_image = image.to(target_device).float()
    working_reference = reference.to(target_device).float()

    if image.dtype == torch.uint8:
        working_image = working_image / 255.0
    if reference.dtype == torch.uint8:
        working_reference = working_reference / 255.0

    matched_images: list[torch.Tensor] = []
    autocast_device = "cuda" if target_device.type == "cuda" else "cpu"

    for index in range(working_image.shape[0]):
        img = working_image[index].permute(2, 0, 1)
        ref = working_reference[index].permute(2, 0, 1)
        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            matched = torch.stack(
                [
                    match_channel(img[0], ref[0], strength, adaptive_matching),
                    match_channel(img[1], ref[1], strength, adaptive_matching),
                    match_channel(img[2], ref[2], strength, adaptive_matching),
                ],
                dim=0,
            ).permute(1, 2, 0)
        matched_images.append(matched)

    return torch.stack(matched_images, dim=0).clamp_(0.0, 1.0).to(image.device)
