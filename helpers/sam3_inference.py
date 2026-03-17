from __future__ import annotations

from contextlib import nullcontext

import comfy.model_management
import torch
from PIL import Image

from .image import pil2tensor, tensor2pil
from .impact_segs import batchify_segments_for_sample, build_seg, extract_resolution
from .masks import compose_masked_result, empty_mask_result
from .segmentation import SEG
from ..lib.sam3 import Sam3Runtime


def select_masks(
    processor,
    img_tensor: torch.Tensor,
    prompt: str,
    confidence_threshold: float,
    max_segments: int,
    segment_pick: int,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    img_pil = tensor2pil(img_tensor)
    state = processor.set_image(img_pil)
    processor.reset_all_prompts(state)
    processor.set_confidence_threshold(confidence_threshold, state)
    state = processor.set_text_prompt(prompt.strip() or "object", state)

    masks = state.get("masks")
    if masks is None or masks.numel() == 0:
        return img_pil, torch.zeros((0, img_pil.height, img_pil.width), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)

    masks = masks.float()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    scores = state.get("scores")
    if scores is None:
        logits = state.get("masks_logits")
        if logits is not None:
            logits = logits.float().squeeze(1) if logits.ndim == 4 else logits.float()
            scores = logits.mean(dim=(-2, -1))
        else:
            scores = torch.ones((masks.shape[0],), dtype=torch.float32, device=masks.device)
    else:
        scores = scores.float().reshape(-1)

    if max_segments > 0 and masks.shape[0] > max_segments:
        topk = torch.topk(scores, k=max_segments)
        masks = masks[topk.indices]
        scores = scores[topk.indices]

    sorted_indices = torch.argsort(scores, descending=True)
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    if segment_pick > 0:
        selected_index = segment_pick - 1
        if selected_index >= masks.shape[0]:
            return img_pil, torch.zeros((0, img_pil.height, img_pil.width), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
        masks = masks[selected_index : selected_index + 1]
        scores = scores[selected_index : selected_index + 1]

    return img_pil, masks, scores


def collect_segmentation_data(
    sam3_model: Sam3Runtime,
    image: torch.Tensor,
    prompt: str,
    confidence_threshold: float,
    max_segments: int,
    segment_pick: int = 0,
) -> list[tuple[torch.Tensor, Image.Image, torch.Tensor, torch.Tensor]]:
    images = image.unsqueeze(0) if image.ndim == 3 else image
    collected: list[tuple[torch.Tensor, Image.Image, torch.Tensor, torch.Tensor]] = []
    autocast_device = comfy.model_management.get_autocast_device(sam3_model.device)
    autocast_enabled = sam3_model.device.type == "cuda" and not comfy.model_management.is_device_mps(sam3_model.device)
    context = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()

    with context:
        for tensor_img in images:
            img_pil, masks, scores = select_masks(sam3_model.processor, tensor_img, prompt, confidence_threshold, max_segments, segment_pick)
            collected.append((tensor_img.detach().cpu(), img_pil, masks.detach().cpu(), scores.detach().cpu()))
    return collected


def render_segmentation_outputs(
    collected,
    output_mode: str,
    mask_blur: int = 0,
    mask_offset: int = 0,
    invert_output: bool = False,
    background: str = "Alpha",
    background_color: str = "#222222",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_mode = (output_mode or "Merged").strip()
    result_images: list[torch.Tensor] = []
    result_masks: list[torch.Tensor] = []
    result_mask_images: list[torch.Tensor] = []

    for _, img_pil, masks, _ in collected:
        if output_mode == "Separate":
            for single_mask in masks:
                result_image, result_mask, mask_rgb = compose_masked_result(
                    img_pil,
                    single_mask,
                    background,
                    background_color,
                    invert_output,
                    mask_blur,
                    mask_offset,
                )
                result_images.append(pil2tensor(result_image))
                result_masks.append(result_mask)
                result_mask_images.append(mask_rgb)
            continue

        if masks.shape[0] == 0:
            result_image, result_mask, mask_rgb = empty_mask_result(img_pil, background, background_color)
        else:
            result_image, result_mask, mask_rgb = compose_masked_result(
                img_pil,
                masks.amax(dim=0),
                background,
                background_color,
                invert_output,
                mask_blur,
                mask_offset,
            )

        result_images.append(pil2tensor(result_image))
        result_masks.append(result_mask)
        result_mask_images.append(mask_rgb)

    if not result_images:
        _, fallback_pil, _, _ = collected[0]
        result_image, result_mask, mask_rgb = empty_mask_result(fallback_pil, background, background_color)
        result_images = [pil2tensor(result_image)]
        result_masks = [result_mask]
        result_mask_images = [mask_rgb]

    return torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0), torch.cat(result_mask_images, dim=0)


def run_segmentation(
    sam3_model: Sam3Runtime,
    image: torch.Tensor,
    prompt: str,
    output_mode: str,
    confidence_threshold: float,
    max_segments: int,
    segment_pick: int = 0,
    mask_blur: int = 0,
    mask_offset: int = 0,
    invert_output: bool = False,
    background: str = "Alpha",
    background_color: str = "#222222",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    collected = collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments, segment_pick)
    return render_segmentation_outputs(collected, output_mode, mask_blur, mask_offset, invert_output, background, background_color)


def build_segments_for_sample(
    image_tensor: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    output_mode: str,
    label: str | None,
    crop_factor: float,
    dilation: int,
) -> list[SEG]:
    output_mode = (output_mode or "Merged").strip()
    if masks.shape[0] == 0:
        return []
    if output_mode == "Merged":
        merged_mask = masks.amax(dim=0)
        confidence = float(scores.max().item()) if scores.numel() > 0 else 1.0
        seg = build_seg(image_tensor, merged_mask, confidence, label, crop_factor, dilation)
        return [seg] if seg is not None else []

    segments: list[SEG] = []
    for index, single_mask in enumerate(masks):
        confidence = float(scores[index].item()) if index < scores.shape[0] else 1.0
        seg = build_seg(image_tensor, single_mask, confidence, label, crop_factor, dilation)
        if seg is not None:
            segments.append(seg)
    return segments


def build_segs_payload(collected, output_mode: str, label: str | None, crop_factor: float, dilation: int) -> tuple[tuple[int, int], list[SEG]]:
    samples = [sample for sample, _, _, _ in collected]
    if not samples:
        return ((64, 64), [])
    resolution = extract_resolution(samples[0])
    detected_segments: list[SEG] = []
    for sample_index, (sample, _, masks, scores) in enumerate(collected):
        sample_segments = build_segments_for_sample(sample, masks, scores, output_mode, label, crop_factor, dilation)
        if len(samples) == 1:
            detected_segments.extend(sample_segments)
        else:
            detected_segments.extend(batchify_segments_for_sample(samples, sample_segments, sample_index))
    return (resolution, detected_segments)


def build_batch_segs_payload(
    collected,
    output_mode: str,
    label: str | None,
    crop_factor: float,
    dilation: int,
) -> list[tuple[tuple[int, int], list[SEG]]]:
    return [
        (extract_resolution(sample), build_segments_for_sample(sample, masks, scores, output_mode, label, crop_factor, dilation))
        for sample, _, masks, scores in collected
    ]