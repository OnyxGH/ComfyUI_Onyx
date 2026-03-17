from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter


def process_mask(mask_image: Image.Image, invert_output: bool = False, mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
    if invert_output:
        mask_array = np.array(mask_image, dtype=np.uint8)
        mask_image = Image.fromarray(255 - mask_array, mode="L")
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        filter_size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filter_type(filter_size))
    return mask_image


def apply_background_color(image: Image.Image, mask_image: Image.Image, background: str = "Alpha", background_color: str = "#222222") -> Image.Image:
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        color = background_color.lstrip("#")
        if len(color) != 6:
            raise ValueError("background_color must be a 6-digit hex color, for example '#222222'.")
        red, green, blue = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        background_image = Image.new("RGBA", image.size, (red, green, blue, 255))
        return Image.alpha_composite(background_image, rgba_image).convert("RGB")
    return rgba_image


def mask_tensor_from_image(mask_image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)


def mask_rgb_from_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    _, height, width = mask_tensor.shape
    return mask_tensor.reshape((1, height, width, 1)).expand(-1, -1, -1, 3)


def empty_mask_result(img_pil: Image.Image, background: str, background_color: str) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    empty_mask_image = Image.new("L", img_pil.size, 0)
    result_image = apply_background_color(img_pil, empty_mask_image, background, background_color)
    result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
    empty_mask = torch.zeros((1, img_pil.height, img_pil.width), dtype=torch.float32)
    return result_image, empty_mask, mask_rgb_from_tensor(empty_mask)


def compose_masked_result(
    img_pil: Image.Image,
    mask_tensor: torch.Tensor,
    background: str,
    background_color: str,
    invert_output: bool,
    mask_blur: int,
    mask_offset: int,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)
    mask_array = (mask_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask_image = process_mask(Image.fromarray(mask_array, mode="L"), invert_output, mask_blur, mask_offset)
    result_image = apply_background_color(img_pil, mask_image, background, background_color)
    result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
    processed_mask = mask_tensor_from_image(mask_image)
    return result_image, processed_mask, mask_rgb_from_tensor(processed_mask)