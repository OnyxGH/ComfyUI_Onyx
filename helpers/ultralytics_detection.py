from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Set

import comfy.samplers
import folder_paths
import numpy as np
import nodes
import torch
import torch.nn.functional as F
from nodes import MAX_RESOLUTION

from .impact_segs import extract_resolution
from .segmentation import (
    SEG,
    build_bbox_mask,
    clamp_bbox,
    dilate_mask,
    ensure_image_batch,
    make_crop_region,
    tensor_to_numpy,
)

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None

_MODEL_CACHE: dict[str, tuple[float, Any]] = {}


def ensure_ultralytics_available() -> None:
    if YOLO is None:
        raise RuntimeError("The 'ultralytics' package is required for this node. Please install it to continue.")


def load_model(model_name: str):
    ensure_ultralytics_available()
    model_path = folder_paths.get_full_path_or_raise("ultralytics", model_name)
    model_mtime = os.path.getmtime(model_path)
    cached = _MODEL_CACHE.get(model_path)
    if cached and cached[0] == model_mtime:
        return cached[1]
    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = (model_mtime, model)
    return model


def resolve_impact_detailer_class():
    node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(node_mappings, dict):
        raise RuntimeError("Comfy node registry is unavailable. Ensure ComfyUI is fully initialized before using this node.")
    detailer_cls = node_mappings.get("DetailerForEach")
    if detailer_cls is None or not hasattr(detailer_cls, "do_detail"):
        raise RuntimeError("Impact Pack's DetailerForEach node is required for this node. Install/enable comfyui-impact-pack.")
    return detailer_cls


def scheduler_options() -> list[str]:
    schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
    try:
        detailer_cls = resolve_impact_detailer_class()
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


def get_model_name_lookup(model) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value).lower() for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value).lower() for index, value in enumerate(names)}
    return {}


def model_supports_segmentation(model) -> bool:
    for task in (
        getattr(model, "task", None),
        getattr(getattr(model, "overrides", None), "get", lambda _key: None)("task") if isinstance(getattr(model, "overrides", None), dict) else None,
        getattr(getattr(model, "model", None), "task", None),
    ):
        if isinstance(task, str) and "seg" in task.lower():
            return True
    return False


def parse_allowed_segments(raw: str, model) -> Optional[Set[int]]:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.replace("\n", ",").split(",")]
    tokens = [token for token in tokens if token]
    if not tokens or (len(tokens) == 1 and tokens[0].lower() == "all"):
        return None

    lookup = get_model_name_lookup(model)
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


def adjust_threshold(current: float, mode: str, step: float) -> float:
    if step <= 0:
        return current
    if (mode or "subtract").lower() == "add":
        return float(min(1.0, current + step))
    return float(max(0.0, current - step))


def format_label(cls_id: Optional[int], label_lookup: Dict[int, str]) -> Optional[str]:
    if cls_id is None:
        return None
    return label_lookup.get(cls_id, str(cls_id))


def finalize_seg(
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


def process_bbox_result(
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
        seg = finalize_seg(image, crop_region, bbox, mask, confidence, format_label(cls_id, label_lookup))
        if seg is not None:
            segments.append(seg)
    return segments


def process_segmentation_result(
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
    class_ids = [int(value) for value in boxes_obj.cls.detach().cpu().tolist()] if boxes_obj is not None and getattr(boxes_obj, "cls", None) is not None else []
    confs = [float(value) for value in boxes_obj.conf.detach().cpu().tolist()] if boxes_obj is not None and getattr(boxes_obj, "conf", None) is not None else []

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
        seg = finalize_seg(image, crop_region, bbox, cropped_mask, confidence, format_label(cls_id, label_lookup))
        if seg is not None:
            segments.append(seg)
    return segments


def detect_and_collect(
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
        return process_segmentation_result(image, result, dilation, crop_factor, drop_size, allowed_ids, label_lookup)
    return process_bbox_result(image, result, dilation, crop_factor, drop_size, allowed_ids, label_lookup)


def detect_with_retries(
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
        detected_segments = detect_and_collect(
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
        threshold = adjust_threshold(threshold, retry_mode, delta)
    return detected_segments


def detect_segments_for_samples(
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
            detect_with_retries(
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


class UltralyticsImpactDetector:
    def __init__(self, model, mode: str, retry_attempts: int = 0, retry_mode: str = "subtract", retry_step: float = 0.05) -> None:
        self.model = model
        self.mode = (mode or "bbox").lower()
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_mode = (retry_mode or "subtract").lower()
        self.retry_step = max(0.0, float(retry_step))
        self.aux: Optional[str] = None
        self.bbox_detector: Optional[UltralyticsImpactDetector] = None

    def detect(self, image: torch.Tensor, threshold: float, dilation: int, crop_factor: float, drop_size: int = 1, detailer_hook=None):
        if self.mode == "segm" and not model_supports_segmentation(self.model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        samples = ensure_image_batch(image)
        if len(samples) != 1:
            raise RuntimeError("[Onyx] Ultralytics Provider detector expects a single image. Use a ForEach/batching node before detection.")

        sample = samples[0]
        segments = detect_with_retries(
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
            parse_allowed_segments(self.aux or "all", self.model),
            get_model_name_lookup(self.model),
        )
        segs = (extract_resolution(sample), segments)
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