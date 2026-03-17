from __future__ import annotations

import torch

from comfy_api.latest import IO

from ..helpers.nodes import get_category, get_node_id
from ..helpers.sam3_inference import build_batch_segs_payload, build_segs_payload, collect_segmentation_data, render_segmentation_outputs, run_segmentation
from ..lib.sam3 import Sam3Runtime, get_sam3_model_options, load_sam3_runtime

CATEGORY = get_category("sam3")


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
    def execute(
        cls,
        sam3_model: Sam3Runtime,
        image: torch.Tensor,
        prompt: str,
        output_mode: str,
        confidence_threshold: float,
        max_segments: int,
        segment_pick: int,
        mask_blur: int,
        mask_offset: int,
        invert_output: bool,
        background: str,
        background_color: str,
    ) -> IO.NodeOutput:
        return IO.NodeOutput(
            *run_segmentation(
                sam3_model,
                image,
                prompt,
                output_mode,
                confidence_threshold,
                max_segments,
                segment_pick,
                mask_blur,
                mask_offset,
                invert_output,
                background,
                background_color,
            )
        )


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
    def execute(
        cls, sam3_model: Sam3Runtime, image: torch.Tensor, prompt: str, output_mode: str, confidence_threshold: float, max_segments: int
    ) -> IO.NodeOutput:
        return IO.NodeOutput(*run_segmentation(sam3_model, image, prompt, output_mode, confidence_threshold, max_segments))


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
    def execute(
        cls,
        sam3_model: Sam3Runtime,
        image: torch.Tensor,
        prompt: str,
        output_mode: str,
        confidence_threshold: float,
        dilation: int,
        crop_factor: float,
        max_segments: int,
    ) -> IO.NodeOutput:
        collected = collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments)
        images, _, _ = render_segmentation_outputs(collected, output_mode)
        segs_payload = build_segs_payload(collected, output_mode, prompt.strip() or None, max(1.0, float(crop_factor)), max(0, int(dilation)))
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
    def execute(
        cls,
        sam3_model: Sam3Runtime,
        image: torch.Tensor,
        prompt: str,
        output_mode: str,
        confidence_threshold: float,
        dilation: int,
        crop_factor: float,
        max_segments: int,
    ) -> IO.NodeOutput:
        collected = collect_segmentation_data(sam3_model, image, prompt, confidence_threshold, max_segments)
        images, _, _ = render_segmentation_outputs(collected, output_mode)
        batch_segs_payload = build_batch_segs_payload(collected, output_mode, prompt.strip() or None, max(1.0, float(crop_factor)), max(0, int(dilation)))
        return IO.NodeOutput(images, batch_segs_payload)


SAM3_NODES: list[type[IO.ComfyNode]] = [
    SAM3ModelLoader,
    SAM3Segment,
    SAM3SegmentSEGS,
    SAM3SegmentSEGSBatch,
    SAM3SegmentRMBG,
]
