from __future__ import annotations

import os
from typing import Any

import comfy.samplers
import folder_paths
import numpy as np
import torch

from comfy_api.latest import IO
from nodes import MAX_RESOLUTION

from ..helpers.impact_segs import batchify_segments_for_sample, extract_resolution, sanitize_segs
from ..helpers.nodes import get_category, get_node_id
from ..helpers.segmentation import (
    SEG,
    empty_image_like,
    ensure_image_batch,
    segments_to_image_list,
)
from ..helpers.ultralytics_detection import (
    UltralyticsImpactDetector,
    detect_segments_for_samples,
    ensure_ultralytics_available,
    get_model_name_lookup,
    load_model,
    model_supports_segmentation,
    parse_allowed_segments,
    resolve_impact_detailer_class,
    scheduler_options,
)
from ..lib.model_paths import add_model_folder_path_ext

CATEGORY = get_category("ultralytics")

add_model_folder_path_ext(
    "ultralytics",
    [os.path.join(folder_paths.models_dir, "ultralytics")],
    folder_paths.supported_pt_extensions,
)


class UltralyticsDetector(IO.ComfyNode):
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
        ensure_ultralytics_available()
        if image is None:
            raise RuntimeError("Image input is required for detection.")

        model = load_model(model_name)
        preference = (model_preference or "bbox").lower()
        if preference not in {"bbox", "segm"}:
            raise ValueError("model_preference must be either 'bbox' or 'segm'.")
        if preference == "segm" and not model_supports_segmentation(model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        crop_factor = max(1.0, float(crop_factor))
        drop_size = int(np.clip(drop_size, 1, MAX_RESOLUTION))
        allowed_ids = parse_allowed_segments(allowed_segments, model)
        samples = ensure_image_batch(image)
        if not samples:
            empty_image = empty_image_like(image)
            fallback_resolution = extract_resolution(image)
            return IO.NodeOutput([empty_image], (fallback_resolution, []))

        label_lookup = get_model_name_lookup(model)
        sample_segments_list = detect_segments_for_samples(
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
                detected_segments.extend(batchify_segments_for_sample(samples, sample_segments, sample_index))

        return IO.NodeOutput(image_list, (extract_resolution(samples[0]), detected_segments))


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
        ensure_ultralytics_available()
        if image is None:
            raise RuntimeError("Image input is required for detection.")

        model = load_model(model_name)
        preference = (model_preference or "bbox").lower()
        if preference not in {"bbox", "segm"}:
            raise ValueError("model_preference must be either 'bbox' or 'segm'.")
        if preference == "segm" and not model_supports_segmentation(model):
            raise RuntimeError("Selected model does not provide segmentation outputs.")

        crop_factor = max(1.0, float(crop_factor))
        drop_size = int(np.clip(drop_size, 1, MAX_RESOLUTION))
        allowed_ids = parse_allowed_segments(allowed_segments, model)
        samples = ensure_image_batch(image)
        if not samples:
            return IO.NodeOutput([empty_image_like(image)], [])

        label_lookup = get_model_name_lookup(model)
        sample_segments_list = detect_segments_for_samples(
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
            batch_segs.append((extract_resolution(sample), sample_segments))
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
                IO.Combo.Input("scheduler", options=scheduler_options()),
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
        detailer_cls = resolve_impact_detailer_class()
        samples = ensure_image_batch(image)
        if len(samples) != 1:
            raise RuntimeError("Detailer (SEGS) expects a single image. Use Detailer (Batch SEGS) for batches.")

        sample_image = samples[0].unsqueeze(0)
        normalized_segs = sanitize_segs(segs, (extract_resolution(samples[0]), []))
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
                IO.Combo.Input("scheduler", options=scheduler_options()),
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
        detailer_cls = resolve_impact_detailer_class()
        samples = ensure_image_batch(image)
        if not samples:
            return IO.NodeOutput(empty_image_like(image), [])

        raw_batch = [batch_segs] if isinstance(batch_segs, tuple) else batch_segs if isinstance(batch_segs, list) else []
        normalized_batch_segs = [
            (sanitize_segs(raw_batch[index], (extract_resolution(sample), [])) if index < len(raw_batch) else (extract_resolution(sample), []))
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
        model = load_model(model_name)
        bbox_detector = UltralyticsImpactDetector(model, "bbox", retry_attempts, retry_mode, retry_step)
        segm_detector = UltralyticsImpactDetector(model, "segm", retry_attempts, retry_mode, retry_step)
        segm_detector.bbox_detector = bbox_detector
        return IO.NodeOutput(bbox_detector, segm_detector)


ULTRALYTICS_NODES: list[type[IO.ComfyNode]] = [
    UltralyticsDetector,
    UltralyticsBatchDetector,
    DetailerForEach,
    DetailerForEachBatch,
    UltralyticsDetectorProvider,
]
