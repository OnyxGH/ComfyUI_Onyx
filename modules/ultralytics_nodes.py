from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Set

import comfy.samplers
import folder_paths
import numpy as np
import nodes
import torch
import torch.nn.functional as F

from comfy_api.latest import IO
from nodes import MAX_RESOLUTION

from ..lib.nodes import get_category, get_node_id
from ..lib.model_paths import add_model_folder_path_ext
from ..lib.segmentation import (
    SEG,
    build_bbox_mask,
    clamp_bbox,
    dilate_mask,
    empty_image_like,
    ensure_image_batch,
    make_crop_region,
    segments_to_image_list,
    tensor_to_numpy,
)

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None


CATEGORY = get_category("ultralytics")

add_model_folder_path_ext(
    "ultralytics",
    [os.path.join(folder_paths.models_dir, "ultralytics")],
    folder_paths.supported_pt_extensions,
)


def _resolve_impact_detailer_class():
    node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(node_mappings, dict):
        raise RuntimeError("Comfy node registry is unavailable. Ensure ComfyUI is fully initialized before using this node.")
    detailer_cls = node_mappings.get("DetailerForEach")
    if detailer_cls is None or not hasattr(detailer_cls, "do_detail"):
        raise RuntimeError("Impact Pack's DetailerForEach node is required for this node. Install/enable comfyui-impact-pack.")
    return detailer_cls


def _scheduler_options() -> list[str]:
    schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
    try:
        detailer_cls = _resolve_impact_detailer_class()
        if hasattr(detailer_cls, "INPUT_TYPES"):
            inputs = detailer_cls.INPUT_TYPES()
            scheduler_input = inputs.get("required", {}).get("scheduler")
            if isinstance(scheduler_input, tuple) and scheduler_input:
                options = scheduler_input[0]
                if isinstance(options, (list, tuple)) and options:
                    return [str(value) for value in options]
    except Exception:
        pass
    return schedulers


def _sanitize_segs(value, fallback: tuple[tuple[int, int], list[SEG]]) -> tuple[tuple[int, int], list[SEG]]:
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


class UltralyticsDetector(IO.ComfyNode):
    _MODEL_CACHE: dict[str, tuple[float, Any]] = {}

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Ultralytics Detector",
            category=CATEGORY,
            description="Detects bounding boxes or segmentation masks with an Ultralytics model and returns cropped images plus SEGS.",
            search_aliases=["ultralytics detector", "yolo detector", "bbox detector", "segmentation detector"],
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("ultralytics")),
                IO.Combo.Input("model_preference", options=["bbox", "segm"]),
                IO.Float.Input("confidence_threshold", default=0.35, min=0.0, max=1.0, step=0.01),
                IO.Int.Input("dilation", default=10, min=0, max=64, step=1),
                IO.Float.Input("crop_factor", default=1.5, min=0.0, max=100.0, step=0.1),
                IO.Int.Input("drop_size", default=10, min=1, max=MAX_RESOLUTION, step=1),
                IO.Int.Input("retry_attempts", default=0, min=0, max=10, step=1),
                IO.Combo.Input("retry_mode", options=["add", "subtract"], default="subtract"),
                IO.Float.Input("retry_step", default=0.05, min=0.0, max=1.0, step=0.01),
                IO.String.Input("allowed_segments", multiline=True, default="all"),
            ],
            outputs=[
                IO.Image.Output("IMAGE_LIST", display_name="IMAGE_LIST", is_output_list=True),
                IO.Custom("SEGS").Output("SEGS", display_name="SEGS"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        model_name: str,
        model_preference: str,
        confidence_threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        retry_attempts: int,
        retry_mode: str,
        retry_step: float,
        allowed_segments: str,
    ) -> IO.NodeOutput:
        if YOLO is None:
            raise RuntimeError("The 'ultralytics' package is required for this node. Please install it to continue.")
        if image is None:
            raise RuntimeError("Image input is required for detection.")

        model = cls._load_model(model_name)
        preference = (model_preference or "bbox").lower()
        if preference not in {"bbox", "segm"}:
            raise ValueError("model_preference must be either 'bbox' or 'segm'.")
        if preference == "segm" and not cls._model_supports_segmentation(model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        crop_factor = max(1.0, float(crop_factor))
        drop_size = int(np.clip(drop_size, 1, MAX_RESOLUTION))
        allowed_ids = cls._parse_allowed_segments(allowed_segments, model)
        samples = ensure_image_batch(image)
        if not samples:
            empty_image = empty_image_like(image)
            fallback_resolution = cls._extract_resolution(image)
            return IO.NodeOutput([empty_image], (fallback_resolution, []))

        label_lookup = cls._get_model_name_lookup(model)
        sample_segments_list = cls._detect_segments_for_samples(
            model,
            samples,
            confidence_threshold,
            retry_attempts,
            retry_mode,
            retry_step,
            dilation,
            preference,
            crop_factor,
            drop_size,
            allowed_ids,
            label_lookup,
        )

        image_list: list[torch.Tensor] = []
        detected_segments: list[SEG] = []
        for sample_index, (sample, sample_segments) in enumerate(zip(samples, sample_segments_list)):
            image_list.extend(segments_to_image_list(sample, sample_segments))
            if len(samples) == 1:
                detected_segments.extend(sample_segments)
            else:
                detected_segments.extend(cls._batchify_segments_for_sample(samples, sample_segments, sample_index))

        return IO.NodeOutput(image_list, (cls._extract_resolution(samples[0]), detected_segments))

    @classmethod
    def _load_model(cls, model_name: str):
        if YOLO is None:
            raise RuntimeError("The 'ultralytics' package is required for this node. Please install it to continue.")
        model_path = folder_paths.get_full_path_or_raise("ultralytics", model_name)
        model_mtime = os.path.getmtime(model_path)
        cached = cls._MODEL_CACHE.get(model_path)
        if cached and cached[0] == model_mtime:
            return cached[1]
        model = YOLO(model_path)
        cls._MODEL_CACHE[model_path] = (model_mtime, model)
        return model

    @staticmethod
    def _extract_resolution(tensor: Optional[torch.Tensor]) -> tuple[int, int]:
        if tensor is None:
            return (64, 64)
        if tensor.ndim == 4:
            return int(tensor.shape[1]), int(tensor.shape[2])
        if tensor.ndim == 3:
            return int(tensor.shape[0]), int(tensor.shape[1])
        return (64, 64)

    @classmethod
    def _detect_segments_for_samples(
        cls,
        model,
        samples: Sequence[torch.Tensor],
        confidence_threshold: float,
        retry_attempts: int,
        retry_mode: str,
        retry_step: float,
        dilation: int,
        mode: str,
        crop_factor: float,
        drop_size: int,
        allowed_ids: Optional[Set[int]],
        label_lookup: Dict[int, str],
    ) -> list[list[SEG]]:
        all_segments: list[list[SEG]] = []
        for sample in samples:
            all_segments.append(
                cls._detect_with_retries(
                    model,
                    sample,
                    confidence_threshold,
                    retry_attempts,
                    retry_mode,
                    retry_step,
                    dilation,
                    mode,
                    crop_factor,
                    drop_size,
                    allowed_ids,
                    label_lookup,
                )
            )
        return all_segments

    @classmethod
    def _detect_with_retries(
        cls,
        model,
        image: torch.Tensor,
        confidence_threshold: float,
        retry_attempts: int,
        retry_mode: str,
        retry_step: float,
        dilation: int,
        mode: str,
        crop_factor: float,
        drop_size: int,
        allowed_ids: Optional[Set[int]],
        label_lookup: Dict[int, str],
    ) -> list[SEG]:
        retries = max(0, int(retry_attempts))
        threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        delta = max(0.0, retry_step)
        detected_segments: list[SEG] = []

        for attempt in range(retries + 1):
            detected_segments = cls._detect_and_collect(
                model,
                image,
                threshold,
                dilation,
                mode,
                crop_factor,
                drop_size,
                allowed_ids,
                label_lookup,
            )
            if detected_segments or attempt == retries:
                break
            threshold = cls._adjust_threshold(threshold, retry_mode, delta)
        return detected_segments

    @classmethod
    def _detect_and_collect(
        cls,
        model,
        image: torch.Tensor,
        threshold: float,
        dilation: int,
        mode: str,
        crop_factor: float,
        drop_size: int,
        allowed_ids: Optional[Set[int]],
        label_lookup: Dict[int, str],
    ) -> list[SEG]:
        np_image = tensor_to_numpy(image)
        results = model([np_image], conf=threshold, verbose=False)
        if isinstance(results, list):
            result = results[0] if results else None
        else:
            try:
                result = list(results)[0]
            except TypeError:
                result = results
        if result is None:
            return []

        if mode == "segm":
            if getattr(result, "masks", None) is None:
                return []
            return cls._process_segmentation_result(image, result, dilation, crop_factor, drop_size, allowed_ids, label_lookup)
        return cls._process_bbox_result(image, result, dilation, crop_factor, drop_size, allowed_ids, label_lookup)

    @staticmethod
    def _process_bbox_result(
        image: torch.Tensor,
        result,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        allowed_ids: Optional[Set[int]],
        label_lookup: Dict[int, str],
    ) -> list[SEG]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else None
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32) if getattr(boxes, "conf", None) is not None else None
        height, width = image.shape[0], image.shape[1]

        segments: list[SEG] = []
        for index, box in enumerate(xyxy):
            cls_id = int(class_ids[index]) if class_ids is not None and index < len(class_ids) else None
            if allowed_ids is not None and cls_id not in allowed_ids:
                continue

            bbox = clamp_bbox(box, width, height)
            if bbox[2] - bbox[0] <= drop_size or bbox[3] - bbox[1] <= drop_size:
                continue

            crop_region = make_crop_region(width, height, bbox, crop_factor)
            mask = build_bbox_mask(crop_region, bbox, dilation)
            confidence = float(confs[index]) if confs is not None and index < len(confs) else 1.0
            seg = UltralyticsDetector._finalize_seg(
                image,
                crop_region,
                bbox,
                mask,
                confidence,
                UltralyticsDetector._format_label(cls_id, label_lookup),
            )
            if seg is not None:
                segments.append(seg)
        return segments

    @classmethod
    def _process_segmentation_result(
        cls,
        image: torch.Tensor,
        result,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        allowed_ids: Optional[Set[int]],
        label_lookup: Dict[int, str],
    ) -> list[SEG]:
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or masks_obj.data is None or len(masks_obj.data) == 0:
            return []

        mask_tensor = masks_obj.data.detach().float().unsqueeze(1)
        mask_tensor = F.interpolate(mask_tensor, size=(image.shape[0], image.shape[1]), mode="bilinear", align_corners=False).squeeze(1)
        mask_tensor = (mask_tensor >= 0.5).float()

        boxes_obj = getattr(result, "boxes", None)
        class_ids = (
            [int(value) for value in boxes_obj.cls.detach().cpu().tolist()] if boxes_obj is not None and getattr(boxes_obj, "cls", None) is not None else []
        )
        confs = (
            [float(value) for value in boxes_obj.conf.detach().cpu().tolist()] if boxes_obj is not None and getattr(boxes_obj, "conf", None) is not None else []
        )

        segments: list[SEG] = []
        for index, mask in enumerate(mask_tensor):
            if dilation > 0:
                mask = dilate_mask(mask, dilation)
            cls_id = class_ids[index] if class_ids and index < len(class_ids) else None
            if allowed_ids is not None and cls_id not in allowed_ids:
                continue
            ys, xs = torch.where(mask > 0.5)
            if ys.numel() == 0 or xs.numel() == 0:
                continue
            bbox = (
                max(int(xs.min().item()), 0),
                max(int(ys.min().item()), 0),
                min(int(xs.max().item()) + 1, image.shape[1]),
                min(int(ys.max().item()) + 1, image.shape[0]),
            )
            if bbox[2] - bbox[0] <= drop_size or bbox[3] - bbox[1] <= drop_size:
                continue
            crop_region = make_crop_region(image.shape[1], image.shape[0], bbox, crop_factor)
            cropped_mask = mask[crop_region[1] : crop_region[3], crop_region[0] : crop_region[2]]
            confidence = confs[index] if index < len(confs) else 1.0
            seg = cls._finalize_seg(image, crop_region, bbox, cropped_mask, confidence, cls._format_label(cls_id, label_lookup))
            if seg is not None:
                segments.append(seg)
        return segments

    @staticmethod
    def _finalize_seg(
        image: torch.Tensor,
        crop_region: tuple[int, int, int, int],
        bbox: tuple[int, int, int, int],
        mask_tensor: torch.Tensor,
        confidence: float,
        label: Optional[str],
    ) -> Optional[SEG]:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2].clone()
        if cropped_image.numel() == 0:
            return None
        mask_np = mask_tensor.detach().cpu().numpy().astype(np.float32)
        if mask_np.size == 0:
            return None
        return SEG(cropped_image, mask_np, float(confidence), crop_region, bbox, label)

    @staticmethod
    def _format_label(cls_id: Optional[int], label_lookup: Dict[int, str]) -> Optional[str]:
        if cls_id is None:
            return None
        return label_lookup.get(cls_id, str(cls_id))

    @staticmethod
    def _adjust_threshold(current: float, mode: str, step: float) -> float:
        if step <= 0:
            return current
        if (mode or "subtract").lower() == "add":
            return float(min(1.0, current + step))
        return float(max(0.0, current - step))

    @classmethod
    def _parse_allowed_segments(cls, raw: str, model) -> Optional[Set[int]]:
        if raw is None:
            return None
        tokens = [token.strip() for token in raw.replace("\n", ",").split(",")]
        tokens = [token for token in tokens if token]
        if not tokens or (len(tokens) == 1 and tokens[0].lower() == "all"):
            return None

        lookup = cls._get_model_name_lookup(model)
        if not lookup:
            return None

        allowed: Set[int] = set()
        for token in tokens:
            token_lower = token.lower()
            if token_lower == "all":
                return None
            if token_lower.isdigit():
                allowed.add(int(token_lower))
                continue
            matches = [index for index, name in lookup.items() if name == token_lower]
            if matches:
                allowed.update(matches)
        return allowed if allowed else set()

    @staticmethod
    def _get_model_name_lookup(model) -> Dict[int, str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(key): str(value).lower() for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value).lower() for index, value in enumerate(names)}
        return {}

    @staticmethod
    def _model_supports_segmentation(model) -> bool:
        for task in (
            getattr(model, "task", None),
            getattr(getattr(model, "overrides", None), "get", lambda _key: None)("task") if isinstance(getattr(model, "overrides", None), dict) else None,
            getattr(getattr(model, "model", None), "task", None),
        ):
            if isinstance(task, str) and "seg" in task.lower():
                return True
        return False

    @staticmethod
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


class UltralyticsBatchDetector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Ultralytics Detector (Batch)",
            category=CATEGORY,
            description="Detects bounding boxes or segmentation masks across an image batch and returns BATCH_SEGS.",
            search_aliases=["ultralytics batch detector", "batch yolo detector", "batch segs detector"],
            inputs=UltralyticsDetector.define_schema().inputs,
            outputs=[
                IO.Image.Output("IMAGE_LIST", display_name="IMAGE_LIST", is_output_list=True),
                IO.Custom("BATCH_SEGS").Output("BATCH_SEGS", display_name="BATCH_SEGS"),
            ],
        )

    @classmethod
    def execute(cls, *args) -> IO.NodeOutput:
        (
            image,
            model_name,
            model_preference,
            confidence_threshold,
            dilation,
            crop_factor,
            drop_size,
            retry_attempts,
            retry_mode,
            retry_step,
            allowed_segments,
        ) = args
        if YOLO is None:
            raise RuntimeError("The 'ultralytics' package is required for this node. Please install it to continue.")
        if image is None:
            raise RuntimeError("Image input is required for detection.")

        model = UltralyticsDetector._load_model(model_name)
        preference = (model_preference or "bbox").lower()
        if preference not in {"bbox", "segm"}:
            raise ValueError("model_preference must be either 'bbox' or 'segm'.")
        if preference == "segm" and not UltralyticsDetector._model_supports_segmentation(model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        crop_factor = max(1.0, float(crop_factor))
        drop_size = int(np.clip(drop_size, 1, MAX_RESOLUTION))
        allowed_ids = UltralyticsDetector._parse_allowed_segments(allowed_segments, model)
        samples = ensure_image_batch(image)
        if not samples:
            return IO.NodeOutput([empty_image_like(image)], [])

        label_lookup = UltralyticsDetector._get_model_name_lookup(model)
        sample_segments_list = UltralyticsDetector._detect_segments_for_samples(
            model,
            samples,
            confidence_threshold,
            retry_attempts,
            retry_mode,
            retry_step,
            dilation,
            preference,
            crop_factor,
            drop_size,
            allowed_ids,
            label_lookup,
        )

        image_list: list[torch.Tensor] = []
        batch_segs: list[tuple[tuple[int, int], list[SEG]]] = []
        for sample, sample_segments in zip(samples, sample_segments_list):
            image_list.extend(segments_to_image_list(sample, sample_segments))
            batch_segs.append((UltralyticsDetector._extract_resolution(sample), sample_segments))
        return IO.NodeOutput(image_list, batch_segs)


class DetailerForEach(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Detailer (SEGS)",
            category=CATEGORY,
            description="Runs Impact Pack's DetailerForEach on a single image with SEGS input.",
            search_aliases=["detailer for each", "detailer segs", "impact detailer"],
            inputs=[
                IO.Image.Input("image"),
                IO.Custom("SEGS").Input("segs"),
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                IO.Vae.Input("vae"),
                IO.Float.Input("guide_size", default=512, min=64, max=MAX_RESOLUTION, step=8),
                IO.Boolean.Input("guide_size_for", default=True),
                IO.Float.Input("max_size", default=1024, min=64, max=MAX_RESOLUTION, step=8),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("steps", default=20, min=1, max=10000),
                IO.Float.Input("cfg", default=8.0, min=0.0, max=100.0),
                IO.Combo.Input("sampler_name", options=list(comfy.samplers.KSampler.SAMPLERS)),
                IO.Combo.Input("scheduler", options=_scheduler_options()),
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Float.Input("denoise", default=0.5, min=0.0001, max=1.0, step=0.01),
                IO.Int.Input("feather", default=5, min=0, max=100, step=1),
                IO.Boolean.Input("noise_mask", default=True),
                IO.Boolean.Input("force_inpaint", default=True),
                IO.String.Input("wildcard", multiline=True, default=""),
                IO.Int.Input("cycle", default=1, min=1, max=10, step=1),
                IO.Boolean.Input("inpaint_model", default=False),
                IO.Int.Input("noise_mask_feather", default=20, min=0, max=100, step=1),
                IO.Boolean.Input("tiled_encode", default=False),
                IO.Boolean.Input("tiled_decode", default=False),
                IO.Custom("DETAILER_HOOK").Input("detailer_hook", optional=True),
                IO.Custom("SCHEDULER_FUNC").Input("scheduler_func_opt", optional=True),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Custom("SEGS").Output("SEGS", display_name="SEGS"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        segs,
        model,
        clip,
        vae,
        guide_size: float,
        guide_size_for: bool,
        max_size: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        denoise: float,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        wildcard: str,
        cycle: int,
        inpaint_model: bool,
        noise_mask_feather: int,
        tiled_encode: bool,
        tiled_decode: bool,
        detailer_hook=None,
        scheduler_func_opt=None,
    ) -> IO.NodeOutput:
        detailer_cls = _resolve_impact_detailer_class()
        samples = ensure_image_batch(image)
        if len(samples) != 1:
            raise RuntimeError("Detailer (SEGS) expects a single image. Use Detailer (Batch SEGS) for batches.")

        sample_image = samples[0].unsqueeze(0)
        normalized_segs = _sanitize_segs(segs, (UltralyticsDetector._extract_resolution(samples[0]), []))
        enhanced_img, *_unused, updated_segs = detailer_cls.do_detail(
            sample_image,
            normalized_segs,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            int(seed),
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            feather,
            noise_mask,
            force_inpaint,
            wildcard,
            detailer_hook,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func_opt=scheduler_func_opt,
            tiled_encode=tiled_encode,
            tiled_decode=tiled_decode,
        )
        return IO.NodeOutput(enhanced_img, updated_segs)


class DetailerForEachBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Detailer (Batch SEGS)",
            category=CATEGORY,
            description="Runs Impact Pack's DetailerForEach across a batch using BATCH_SEGS input.",
            search_aliases=["detailer batch segs", "batch detailer", "impact batch detailer"],
            inputs=[
                IO.Image.Input("image"),
                IO.Custom("BATCH_SEGS").Input("batch_segs"),
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                IO.Vae.Input("vae"),
                IO.Float.Input("guide_size", default=512, min=64, max=MAX_RESOLUTION, step=8),
                IO.Boolean.Input("guide_size_for", default=True),
                IO.Float.Input("max_size", default=1024, min=64, max=MAX_RESOLUTION, step=8),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("steps", default=20, min=1, max=10000),
                IO.Float.Input("cfg", default=8.0, min=0.0, max=100.0),
                IO.Combo.Input("sampler_name", options=list(comfy.samplers.KSampler.SAMPLERS)),
                IO.Combo.Input("scheduler", options=_scheduler_options()),
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Float.Input("denoise", default=0.5, min=0.0001, max=1.0, step=0.01),
                IO.Int.Input("feather", default=5, min=0, max=100, step=1),
                IO.Boolean.Input("noise_mask", default=True),
                IO.Boolean.Input("force_inpaint", default=True),
                IO.String.Input("wildcard", multiline=True, default=""),
                IO.Int.Input("cycle", default=1, min=1, max=10, step=1),
                IO.Boolean.Input("inpaint_model", default=False),
                IO.Int.Input("noise_mask_feather", default=20, min=0, max=100, step=1),
                IO.Boolean.Input("tiled_encode", default=False),
                IO.Boolean.Input("tiled_decode", default=False),
                IO.Custom("DETAILER_HOOK").Input("detailer_hook", optional=True),
                IO.Custom("SCHEDULER_FUNC").Input("scheduler_func_opt", optional=True),
            ],
            outputs=[
                IO.Image.Output("IMAGE", display_name="IMAGE"),
                IO.Custom("BATCH_SEGS").Output("BATCH_SEGS", display_name="BATCH_SEGS"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        batch_segs,
        model,
        clip,
        vae,
        guide_size: float,
        guide_size_for: bool,
        max_size: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        denoise: float,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        wildcard: str,
        cycle: int,
        inpaint_model: bool,
        noise_mask_feather: int,
        tiled_encode: bool,
        tiled_decode: bool,
        detailer_hook=None,
        scheduler_func_opt=None,
    ) -> IO.NodeOutput:
        detailer_cls = _resolve_impact_detailer_class()
        samples = ensure_image_batch(image)
        if not samples:
            return IO.NodeOutput(empty_image_like(image), [])

        raw_batch = [batch_segs] if isinstance(batch_segs, tuple) else batch_segs if isinstance(batch_segs, list) else []
        normalized_batch_segs = [
            (
                _sanitize_segs(raw_batch[index], (UltralyticsDetector._extract_resolution(sample), []))
                if index < len(raw_batch)
                else (UltralyticsDetector._extract_resolution(sample), [])
            )
            for index, sample in enumerate(samples)
        ]

        enhanced_images: list[torch.Tensor] = []
        new_batch_segs: list[tuple[tuple[int, int], list[SEG]]] = []
        for batch_index, (sample, sample_segs) in enumerate(zip(samples, normalized_batch_segs)):
            enhanced_img, *_unused, updated_segs = detailer_cls.do_detail(
                sample.unsqueeze(0),
                sample_segs,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for,
                max_size,
                int(seed) + batch_index,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                denoise,
                feather,
                noise_mask,
                force_inpaint,
                wildcard,
                detailer_hook,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt,
                tiled_encode=tiled_encode,
                tiled_decode=tiled_decode,
            )
            enhanced_images.append(enhanced_img)
            new_batch_segs.append(updated_segs)

        enhanced_batch = torch.cat(enhanced_images, dim=0) if enhanced_images else empty_image_like(image)
        return IO.NodeOutput(enhanced_batch, new_batch_segs)


class _UltralyticsImpactDetector:
    def __init__(self, model, mode: str, retry_attempts: int = 0, retry_mode: str = "subtract", retry_step: float = 0.05) -> None:
        self.model = model
        self.mode = (mode or "bbox").lower()
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_mode = (retry_mode or "subtract").lower()
        self.retry_step = max(0.0, float(retry_step))
        self.aux: Optional[str] = None
        self.bbox_detector: Optional[_UltralyticsImpactDetector] = None

    def detect(self, image: torch.Tensor, threshold: float, dilation: int, crop_factor: float, drop_size: int = 1, detailer_hook=None):
        if self.mode == "segm" and not UltralyticsDetector._model_supports_segmentation(self.model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        samples = ensure_image_batch(image)
        if len(samples) != 1:
            raise RuntimeError("[Onyx] Ultralytics Provider detector expects a single image. Use a ForEach/batching node before detection.")

        sample = samples[0]
        segments = UltralyticsDetector._detect_with_retries(
            self.model,
            sample,
            float(np.clip(threshold, 0.0, 1.0)),
            self.retry_attempts,
            self.retry_mode,
            self.retry_step,
            int(dilation),
            self.mode,
            max(1.0, float(crop_factor)),
            int(np.clip(drop_size, 1, MAX_RESOLUTION)),
            UltralyticsDetector._parse_allowed_segments(self.aux or "all", self.model),
            UltralyticsDetector._get_model_name_lookup(self.model),
        )
        segs = (UltralyticsDetector._extract_resolution(sample), segments)
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs

    def detect_combined(self, image: torch.Tensor, threshold: float, dilation: int) -> torch.Tensor:
        segs = self.detect(image, threshold, dilation, 1.0, 1, detailer_hook=None)
        height, width = segs[0]
        mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        for seg in segs[1]:
            crop_x1, crop_y1, crop_x2, crop_y2 = seg.crop_region
            crop_w = max(0, crop_x2 - crop_x1)
            crop_h = max(0, crop_y2 - crop_y1)
            if crop_w <= 0 or crop_h <= 0:
                continue
            seg_mask = np.asarray(seg.cropped_mask, dtype=np.float32)
            if seg_mask.ndim == 3:
                seg_mask = np.max(seg_mask, axis=0)
            if seg_mask.ndim != 2 or seg_mask.shape != (crop_h, crop_w):
                continue
            seg_mask_t = torch.from_numpy(np.clip(seg_mask, 0.0, 1.0))
            mask[crop_y1:crop_y2, crop_x1:crop_x2] = torch.maximum(mask[crop_y1:crop_y2, crop_x1:crop_x2], seg_mask_t)
        return mask

    def setAux(self, x) -> None:
        token = str(x).strip() if x is not None else ""
        self.aux = token or None


class UltralyticsDetectorProvider(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Ultralytics Provider",
            category=CATEGORY,
            description="Creates Impact Pack-compatible bbox and segmentation detectors backed by an Ultralytics model.",
            search_aliases=["ultralytics provider", "yolo provider", "detector provider"],
            inputs=[
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("ultralytics")),
                IO.Int.Input("retry_attempts", default=0, min=0, max=10, step=1),
                IO.Combo.Input("retry_mode", options=["add", "subtract"], default="subtract"),
                IO.Float.Input("retry_step", default=0.05, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Custom("BBOX_DETECTOR").Output("BBOX_DETECTOR", display_name="BBOX_DETECTOR"),
                IO.Custom("SEGM_DETECTOR").Output("SEGM_DETECTOR", display_name="SEGM_DETECTOR"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, retry_attempts: int, retry_mode: str, retry_step: float) -> IO.NodeOutput:
        model = UltralyticsDetector._load_model(model_name)
        bbox_detector = _UltralyticsImpactDetector(model, "bbox", retry_attempts, retry_mode, retry_step)
        segm_detector = _UltralyticsImpactDetector(model, "segm", retry_attempts, retry_mode, retry_step)
        segm_detector.bbox_detector = bbox_detector
        return IO.NodeOutput(bbox_detector, segm_detector)


ULTRALYTICS_NODES: list[type[IO.ComfyNode]] = [
    UltralyticsDetector,
    UltralyticsBatchDetector,
    DetailerForEach,
    DetailerForEachBatch,
    UltralyticsDetectorProvider,
]
