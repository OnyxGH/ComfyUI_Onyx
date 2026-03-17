from __future__ import annotations

import json

import comfy.samplers
import torch

from PIL.PngImagePlugin import PngInfo

from comfy.cli_args import args
from comfy_api.latest import IO, UI

from ..lib.bundle import normalize_bundle

IMAGE_OUTPUT_TYPE = "output"
IMAGE_COMPRESS_LEVEL = 4


def output_image(image: torch.Tensor, mode: str, filename_prefix: str, cls: type[IO.ComfyNode]) -> IO.NodeOutput:
    if mode == "preview_only":
        return IO.NodeOutput(ui=UI.PreviewImage(image, cls=cls))
    if mode == "save":
        return IO.NodeOutput(ui=UI.ImageSaveHelper.get_save_images_ui(images=image, filename_prefix=filename_prefix, cls=cls))
    raise ValueError(f"Unsupported mode for outputting image: {mode}")


def output_image_bundle(bundle, mode: str, filename_prefix: str, cls: type[IO.ComfyNode]) -> IO.NodeOutput:
    normalized_bundle = normalize_bundle(bundle)
    if "image" not in normalized_bundle:
        raise ValueError("The bundle must include an image for output.")
    image = normalized_bundle["image"]
    return output_image(image, mode, filename_prefix, cls=cls)


def format_a1111_parameters(
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    sampler: str,
    scheduler: str,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    denoise: float,
    checkpoint: str,
) -> str:
    positive = "" if positive_prompt is None else str(positive_prompt)
    negative = "" if negative_prompt is None else str(negative_prompt)

    parts = [
        f"Steps: {steps}",
        f"Sampler: {sampler}",
        f"Scheduler: {scheduler}",
        f"CFG scale: {cfg}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
    ]
    if checkpoint:
        parts.append(f"Model: {checkpoint}")
    if denoise is not None:
        parts.append(f"Denoising strength: {denoise}")

    return f"{positive}\nNegative prompt: {negative}\n{', '.join(parts)}"


def build_basic_workflow(
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    checkpoint: str,
    width: int,
    height: int,
) -> dict:
    sampler_name = sampler or comfy.samplers.KSampler.SAMPLERS[0]
    scheduler_name = scheduler or comfy.samplers.KSampler.SCHEDULERS[0]
    checkpoint_name = checkpoint or ""

    return {
        "last_node_id": 7,
        "last_link_id": 9,
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "pos": [20, 470],
                "size": [315, 98],
                "flags": {},
                "order": 0,
                "mode": 0,
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
                    {"name": "CLIP", "type": "CLIP", "links": [2, 4], "slot_index": 1},
                    {"name": "VAE", "type": "VAE", "links": [8], "slot_index": 2},
                ],
                "properties": {},
                "widgets_values": [checkpoint_name],
            },
            {
                "id": 2,
                "type": "CLIPTextEncode",
                "pos": [410, 180],
                "size": [425, 164],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [{"name": "clip", "type": "CLIP", "link": 2}],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [3], "slot_index": 0}],
                "properties": {},
                "widgets_values": [positive_prompt or ""],
            },
            {
                "id": 3,
                "type": "CLIPTextEncode",
                "pos": [410, 380],
                "size": [425, 164],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5], "slot_index": 0}],
                "properties": {},
                "widgets_values": [negative_prompt or ""],
            },
            {
                "id": 4,
                "type": "EmptyLatentImage",
                "pos": [470, 610],
                "size": [315, 106],
                "flags": {},
                "order": 3,
                "mode": 0,
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6], "slot_index": 0}],
                "properties": {},
                "widgets_values": [int(width), int(height), 1],
            },
            {
                "id": 5,
                "type": "KSampler",
                "pos": [860, 180],
                "size": [315, 262],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                    {"name": "positive", "type": "CONDITIONING", "link": 3},
                    {"name": "negative", "type": "CONDITIONING", "link": 5},
                    {"name": "latent_image", "type": "LATENT", "link": 6},
                ],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7], "slot_index": 0}],
                "properties": {},
                "widgets_values": [int(seed), True, int(steps), float(cfg), sampler_name, scheduler_name, float(denoise)],
            },
            {
                "id": 6,
                "type": "VAEDecode",
                "pos": [1210, 180],
                "size": [210, 46],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 7},
                    {"name": "vae", "type": "VAE", "link": 8},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}],
                "properties": {},
            },
            {
                "id": 7,
                "type": "SaveImage",
                "pos": [1450, 180],
                "size": [210, 26],
                "flags": {},
                "order": 6,
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
                "properties": {},
            },
        ],
        "links": [
            [1, 1, 0, 5, 0, "MODEL"],
            [2, 1, 1, 2, 0, "CLIP"],
            [3, 2, 0, 5, 1, "CONDITIONING"],
            [4, 1, 1, 3, 0, "CLIP"],
            [5, 3, 0, 5, 2, "CONDITIONING"],
            [6, 4, 0, 5, 3, "LATENT"],
            [7, 5, 0, 6, 0, "LATENT"],
            [8, 1, 2, 6, 1, "VAE"],
            [9, 6, 0, 7, 0, "IMAGE"],
        ],
        "groups": [],
        "config": {},
        "extra": {"ds": {"offset": [0, 0], "scale": 1}},
        "version": 0.4,
    }


def resolve_hidden_metadata(cls: type[IO.ComfyNode], prompt, extra_pnginfo) -> tuple[object | None, object | None]:
    hidden = getattr(cls, "hidden", None)
    resolved_prompt = prompt if prompt is not None else getattr(hidden, "prompt", None) if hidden is not None else None
    resolved_extra_pnginfo = (
        extra_pnginfo if extra_pnginfo is not None else getattr(hidden, "extra_pnginfo", None) if hidden is not None else None
    )
    return resolved_prompt, resolved_extra_pnginfo


def coerce_extra_pnginfo_dict(extra_pnginfo) -> dict | None:
    if isinstance(extra_pnginfo, dict):
        return extra_pnginfo
    if isinstance(extra_pnginfo, str):
        try:
            parsed = json.loads(extra_pnginfo)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def resolve_workflow_text(
    *,
    basic_workflow: bool,
    prompt,
    extra_pnginfo,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    checkpoint: str,
    width: int,
    height: int,
    original_width,
    original_height,
) -> tuple[str | None, str | None]:
    prompt_text = None
    if not basic_workflow and prompt is not None:
        prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt)

    if basic_workflow:
        workflow_width = int(original_width) if original_width is not None else width
        workflow_height = int(original_height) if original_height is not None else height
        workflow_text = json.dumps(
            build_basic_workflow(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                denoise=denoise,
                checkpoint=checkpoint,
                width=workflow_width,
                height=workflow_height,
            )
        )
        return prompt_text, workflow_text

    extra_pnginfo_dict = coerce_extra_pnginfo_dict(extra_pnginfo)
    if not isinstance(extra_pnginfo_dict, dict):
        return prompt_text, None

    workflow = extra_pnginfo_dict.get("workflow")
    if workflow is None:
        workflow = extra_pnginfo_dict.get("Workflow")
    if workflow is None:
        return prompt_text, None

    workflow_text = workflow if isinstance(workflow, str) else json.dumps(workflow)
    return prompt_text, workflow_text


def build_save_image_advanced_metadata(
    *,
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    checkpoint: str,
    original_width,
    original_height,
    embed_workflow: bool,
    basic_workflow: bool,
    prompt,
    extra_pnginfo,
) -> PngInfo | None:
    if args.disable_metadata:
        return None

    width = int(image.shape[1])
    height = int(image.shape[0])
    metadata = PngInfo()
    metadata.add_text(
        "parameters",
        format_a1111_parameters(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            sampler=sampler or "",
            scheduler=scheduler or "",
            cfg=cfg,
            seed=seed,
            width=width,
            height=height,
            denoise=denoise,
            checkpoint=checkpoint or "",
        ),
    )

    if not embed_workflow:
        return metadata

    resolved_prompt, resolved_extra_pnginfo = resolve_hidden_metadata(cls, prompt, extra_pnginfo)
    prompt_text, workflow_text = resolve_workflow_text(
        basic_workflow=basic_workflow,
        prompt=resolved_prompt,
        extra_pnginfo=resolved_extra_pnginfo,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        denoise=denoise,
        checkpoint=checkpoint,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
    )
    if prompt_text is not None:
        metadata.add_text("prompt", prompt_text)
    if workflow_text is not None:
        metadata.add_text("workflow", workflow_text)

    return metadata