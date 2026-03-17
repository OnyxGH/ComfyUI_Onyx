from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter

import comfy.model_management

from comfy_api.latest import IO

from ..lib.nodes import get_category, get_node_id
from ..lib.images import pil2tensor, tensor2pil
from ..lib.sam3 import Sam3Runtime, get_sam3_model_options, load_sam3_runtime
from ..lib.segmentation import SEG, dilate_mask, make_crop_region

CATEGORY = get_category("sam3")


def _process_mask(mask_image: Image.Image, invert_output: bool = False, mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
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


def _apply_background_color(image: Image.Image, mask_image: Image.Image, background: str = "Alpha", background_color: str = "#222222") -> Image.Image:
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


def _mask_tensor_from_image(mask_image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)


def _mask_rgb_from_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    _, height, width = mask_tensor.shape
    return mask_tensor.reshape((1, height, width, 1)).expand(-1, -1, -1, 3)


def _empty_result(img_pil: Image.Image, background: str, background_color: str) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    empty_mask_image = Image.new("L", img_pil.size, 0)
    result_image = _apply_background_color(img_pil, empty_mask_image, background, background_color)
    result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
    empty_mask = torch.zeros((1, img_pil.height, img_pil.width), dtype=torch.float32)
    return result_image, empty_mask, _mask_rgb_from_tensor(empty_mask)


def _select_masks(processor, img_tensor: torch.Tensor, prompt: str, confidence_threshold: float, max_segments: int, segment_pick: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
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
        masks = masks[selected_index:selected_index + 1]
        scores = scores[selected_index:selected_index + 1]

    return img_pil, masks, scores


def _compose_result(img_pil: Image.Image, mask_tensor: torch.Tensor, background: str, background_color: str, invert_output: bool, mask_blur: int, mask_offset: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)
    mask_array = (mask_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask_image = _process_mask(Image.fromarray(mask_array, mode="L"), invert_output, mask_blur, mask_offset)
    result_image = _apply_background_color(img_pil, mask_image, background, background_color)
    result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
    processed_mask = _mask_tensor_from_image(mask_image)
    return result_image, processed_mask, _mask_rgb_from_tensor(processed_mask)


def _collect_segmentation_data(sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, confidence_threshold: float, max_segments: int, segment_pick: int = 0) -> list[tuple[torch.Tensor, Image.Image, torch.Tensor, torch.Tensor]]:
    images = image.unsqueeze(0) if image.ndim == 3 else image
    collected: list[tuple[torch.Tensor, Image.Image, torch.Tensor, torch.Tensor]] = []
    autocast_device = comfy.model_management.get_autocast_device(sam3_model.device)
    autocast_enabled = sam3_model.device.type == "cuda" and not comfy.model_management.is_device_mps(sam3_model.device)
    context = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()

    with context:
        for tensor_img in images:
            img_pil, masks, scores = _select_masks(sam3_model.processor, tensor_img, prompt, confidence_threshold, max_segments, segment_pick)
            collected.append((tensor_img.detach().cpu(), img_pil, masks.detach().cpu(), scores.detach().cpu()))
    return collected


def _render_segmentation_outputs(collected, output_mode: str, mask_blur: int = 0, mask_offset: int = 0, invert_output: bool = False, background: str = "Alpha", background_color: str = "#222222") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_mode = (output_mode or "Merged").strip()
    result_images: list[torch.Tensor] = []
    result_masks: list[torch.Tensor] = []
    result_mask_images: list[torch.Tensor] = []

    for _, img_pil, masks, _ in collected:
        if output_mode == "Separate":
            for single_mask in masks:
                result_image, result_mask, mask_rgb = _compose_result(img_pil, single_mask, background, background_color, invert_output, mask_blur, mask_offset)
                result_images.append(pil2tensor(result_image))
                result_masks.append(result_mask)
                result_mask_images.append(mask_rgb)
            continue

        if masks.shape[0] == 0:
            result_image, result_mask, mask_rgb = _empty_result(img_pil, background, background_color)
        else:
            result_image, result_mask, mask_rgb = _compose_result(img_pil, masks.amax(dim=0), background, background_color, invert_output, mask_blur, mask_offset)

        result_images.append(pil2tensor(result_image))
        result_masks.append(result_mask)
        result_mask_images.append(mask_rgb)

    if not result_images:
        _, fallback_pil, _, _ = collected[0]
        result_image, result_mask, mask_rgb = _empty_result(fallback_pil, background, background_color)
        result_images = [pil2tensor(result_image)]
        result_masks = [result_mask]
        result_mask_images = [mask_rgb]

    return torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0), torch.cat(result_mask_images, dim=0)


def _run_segmentation(sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, max_segments: int, segment_pick: int = 0, mask_blur: int = 0, mask_offset: int = 0, invert_output: bool = False, background: str = "Alpha", background_color: str = "#222222") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    collected = _collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments, segment_pick)
    return _render_segmentation_outputs(collected, output_mode, mask_blur, mask_offset, invert_output, background, background_color)


def _extract_resolution(tensor: torch.Tensor | None) -> tuple[int, int]:
    if tensor is None:
        return (64, 64)
    if tensor.ndim == 4:
        return int(tensor.shape[1]), int(tensor.shape[2])
    if tensor.ndim == 3:
        return int(tensor.shape[0]), int(tensor.shape[1])
    return (64, 64)


def _build_seg(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, confidence: float, label: str | None, crop_factor: float, dilation: int) -> SEG | None:
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


def _build_segments_for_sample(image_tensor: torch.Tensor, masks: torch.Tensor, scores: torch.Tensor, output_mode: str, label: str | None, crop_factor: float, dilation: int) -> list[SEG]:
    output_mode = (output_mode or "Merged").strip()
    if masks.shape[0] == 0:
        return []
    if output_mode == "Merged":
        merged_mask = masks.amax(dim=0)
        confidence = float(scores.max().item()) if scores.numel() > 0 else 1.0
        seg = _build_seg(image_tensor, merged_mask, confidence, label, crop_factor, dilation)
        return [seg] if seg is not None else []

    segments: list[SEG] = []
    for index, single_mask in enumerate(masks):
        confidence = float(scores[index].item()) if index < scores.shape[0] else 1.0
        seg = _build_seg(image_tensor, single_mask, confidence, label, crop_factor, dilation)
        if seg is not None:
            segments.append(seg)
    return segments


def _batchify_segments_for_sample(samples: list[torch.Tensor], segments: list[SEG], sample_index: int) -> list[SEG]:
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


def _build_segs_payload(collected, output_mode: str, label: str | None, crop_factor: float, dilation: int) -> tuple[tuple[int, int], list[SEG]]:
    samples = [sample for sample, _, _, _ in collected]
    if not samples:
        return ((64, 64), [])
    resolution = _extract_resolution(samples[0])
    detected_segments: list[SEG] = []
    for sample_index, (sample, _, masks, scores) in enumerate(collected):
        sample_segments = _build_segments_for_sample(sample, masks, scores, output_mode, label, crop_factor, dilation)
        if len(samples) == 1:
            detected_segments.extend(sample_segments)
        else:
            detected_segments.extend(_batchify_segments_for_sample(samples, sample_segments, sample_index))
    return (resolution, detected_segments)


def _build_batch_segs_payload(collected, output_mode: str, label: str | None, crop_factor: float, dilation: int) -> list[tuple[tuple[int, int], list[SEG]]]:
    return [(_extract_resolution(sample), _build_segments_for_sample(sample, masks, scores, output_mode, label, crop_factor, dilation)) for sample, _, masks, scores in collected]


class SAM3ModelLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Load SAM3 Model",
            category=CATEGORY,
            description="Loads a SAM3 checkpoint and returns a reusable SAM3 model object for segmentation nodes.",
            search_aliases=["load sam3 model", "sam3 loader", "sam3 model loader"],
            inputs=[
                IO.Combo.Input("model_name", options=get_sam3_model_options()),
                IO.Combo.Input("device", options=["Auto", "CPU", "GPU"], default="Auto"),
            ],
            outputs=[IO.Custom("SAM3_MODEL").Output("SAM3_MODEL", display_name="SAM3_MODEL")],
        )

    @classmethod
    def execute(cls, model_name: str, device: str) -> IO.NodeOutput:
        return IO.NodeOutput(load_sam3_runtime(model_name=model_name, device_choice=device))


class SAM3SegmentRMBG(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SAM3 Segment (RMBG-like)",
            category=CATEGORY,
            description="Segments images from a text prompt using a loaded SAM3 model.",
            search_aliases=["sam3 rmbg", "sam3 segment rmbg", "sam3 background removal"],
            inputs=[
                IO.Custom("SAM3_MODEL").Input("sam3_model", display_name="sam3_model"),
                IO.Image.Input("image"),
                IO.String.Input("prompt", default="", multiline=True, placeholder="Describe the concept"),
                IO.Combo.Input("output_mode", options=["Merged", "Separate"], default="Merged"),
                IO.Float.Input("confidence_threshold", default=0.5, min=0.05, max=0.95, step=0.01),
                IO.Int.Input("max_segments", default=0, min=0, max=128, step=1),
                IO.Int.Input("segment_pick", default=0, min=0, max=128, step=1),
                IO.Int.Input("mask_blur", default=0, min=0, max=64, step=1),
                IO.Int.Input("mask_offset", default=0, min=-64, max=64, step=1),
                IO.Boolean.Input("invert_output", default=False),
                IO.Combo.Input("background", options=["Alpha", "Color"], default="Alpha"),
                IO.Color.Input("background_color", default="#222222"),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Mask.Output("MASK", display_name="MASK"),
                IO.Image.Output("MASK_IMAGE", display_name="MASK_IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, max_segments: int, segment_pick: int, mask_blur: int, mask_offset: int, invert_output: bool, background: str, background_color: str) -> IO.NodeOutput:
        return IO.NodeOutput(*_run_segmentation(sam3_model, image, prompt, output_mode, confidence_threshold, max_segments, segment_pick, mask_blur, mask_offset, invert_output, background, background_color))


class SAM3Segment(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SAM3 Segment",
            category=CATEGORY,
            description="Segments images from a text prompt using a loaded SAM3 model.",
            search_aliases=["sam3 segment", "sam3 segmentation", "segment with sam3"],
            inputs=[
                IO.Custom("SAM3_MODEL").Input("sam3_model", display_name="sam3_model"),
                IO.Image.Input("image"),
                IO.String.Input("prompt", default="", placeholder="Describe the concept"),
                IO.Combo.Input("output_mode", options=["Merged", "Separate"], default="Merged"),
                IO.Float.Input("confidence_threshold", default=0.5, min=0.05, max=0.95, step=0.01),
                IO.Int.Input("max_segments", default=0, min=0, max=128, step=1),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Mask.Output("MASK", display_name="MASK"),
                IO.Image.Output("MASK_IMAGE", display_name="MASK_IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, max_segments: int) -> IO.NodeOutput:
        return IO.NodeOutput(*_run_segmentation(sam3_model, image, prompt, output_mode, confidence_threshold, max_segments))


class SAM3SegmentSEGS(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SAM3 Segment (SEGS)",
            category=CATEGORY,
            description="Segments images from a text prompt using a loaded SAM3 model and returns Impact Pack-compatible SEGS.",
            search_aliases=["sam3 segs", "sam3 segment segs", "sam3 impact segs"],
            inputs=[
                IO.Custom("SAM3_MODEL").Input("sam3_model", display_name="sam3_model"),
                IO.Image.Input("image"),
                IO.String.Input("prompt", default="", placeholder="Describe the concept"),
                IO.Combo.Input("output_mode", options=["Merged", "Separate"], default="Merged"),
                IO.Float.Input("confidence_threshold", default=0.5, min=0.05, max=0.95, step=0.01),
                IO.Int.Input("dilation", default=0, min=0, max=64, step=1),
                IO.Float.Input("crop_factor", default=1.5, min=0.0, max=100.0, step=0.1),
                IO.Int.Input("max_segments", default=0, min=0, max=128, step=1),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Custom("SEGS").Output("SEGS", display_name="SEGS"),
            ],
        )

    @classmethod
    def execute(cls, sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, dilation: int, crop_factor: float, max_segments: int) -> IO.NodeOutput:
        collected = _collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments)
        images, _, _ = _render_segmentation_outputs(collected, output_mode)
        segs_payload = _build_segs_payload(collected, output_mode, prompt.strip() or None, max(1.0, float(crop_factor)), max(0, int(dilation)))
        return IO.NodeOutput(images, segs_payload)


class SAM3SegmentSEGSBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SAM3 Segment (Batch SEGS)",
            category=CATEGORY,
            description="Segments images from a text prompt using a loaded SAM3 model and returns this project's BATCH_SEGS format.",
            search_aliases=["sam3 batch segs", "sam3 segment batch segs", "sam3 batch segmentation"],
            inputs=[
                IO.Custom("SAM3_MODEL").Input("sam3_model", display_name="sam3_model"),
                IO.Image.Input("image"),
                IO.String.Input("prompt", default="", placeholder="Describe the concept"),
                IO.Combo.Input("output_mode", options=["Merged", "Separate"], default="Merged"),
                IO.Float.Input("confidence_threshold", default=0.5, min=0.05, max=0.95, step=0.01),
                IO.Int.Input("dilation", default=0, min=0, max=64, step=1),
                IO.Float.Input("crop_factor", default=1.5, min=0.0, max=100.0, step=0.1),
                IO.Int.Input("max_segments", default=0, min=0, max=128, step=1),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Custom("BATCH_SEGS").Output("BATCH_SEGS", display_name="BATCH_SEGS"),
            ],
        )

    @classmethod
    def execute(cls, sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, dilation: int, crop_factor: float, max_segments: int) -> IO.NodeOutput:
        collected = _collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments)
        images, _, _ = _render_segmentation_outputs(collected, output_mode)
        batch_segs_payload = _build_batch_segs_payload(collected, output_mode, prompt.strip() or None, max(1.0, float(crop_factor)), max(0, int(dilation)))
        return IO.NodeOutput(images, batch_segs_payload)


SAM3_NODES: list[type[IO.ComfyNode]] = [
    SAM3ModelLoader,
    SAM3Segment,
    SAM3SegmentSEGS,
    SAM3SegmentSEGSBatch,
    SAM3SegmentRMBG,
]