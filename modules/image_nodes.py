from __future__ import annotations

import os
import numpy as np
import torch

import folder_paths
import nodes
import comfy.samplers

from PIL import Image
from comfy_api.latest import IO

from ..lib.bundle import BundleType, normalize_bundle, update_bundle
from ..helpers.color import MATCH_COLOR_METHODS, match_color
from ..helpers.nodes import get_category, get_node_id
from ..helpers.output import (
    IMAGE_COMPRESS_LEVEL,
    IMAGE_OUTPUT_TYPE,
    build_save_image_advanced_metadata,
    output_image,
    output_image_bundle,
    resolve_hidden_metadata,
)

CATEGORY = get_category("image")


class SaveImageAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save Image Advanced",
            category=CATEGORY,
            description="Saves the input images to your ComfyUI output directory.",
            search_aliases=["save", "save image", "export image", "output image", "write image", "download"],
            is_output_node=True,
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.String.Input("positive_prompt", display_name="positive", multiline=True, dynamic_prompts=True, default=""),
                IO.String.Input("negative_prompt", display_name="negative", multiline=True, dynamic_prompts=True, default=""),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("steps", default=20, min=1, max=10000),
                IO.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
                IO.Combo.Input("sampler", options=comfy.samplers.KSampler.SAMPLERS, default=comfy.samplers.KSampler.SAMPLERS[0]),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default=comfy.samplers.KSampler.SCHEDULERS[0]),
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, round=0.001),
                IO.Combo.Input("checkpoint", options=folder_paths.get_filename_list("checkpoints")),
                IO.Int.Input("original_width", optional=True, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("original_height", optional=True, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Boolean.Input("embed_workflow", default=False),
                IO.Boolean.Input("basic_workflow", default=False),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            essentials_category="Basics",
        )

    @classmethod
    def execute(
        cls,
        images,
        filename_prefix="ComfyUI",
        positive_prompt="",
        negative_prompt="",
        seed=0,
        steps=20,
        cfg=7.0,
        sampler=None,
        scheduler=None,
        denoise=1.0,
        checkpoint="",
        original_width=None,
        original_height=None,
        embed_workflow=False,
        basic_workflow=False,
        prompt=None,
        extra_pnginfo=None,
    ) -> IO.NodeOutput:
        if images is None or len(images) == 0:
            return IO.NodeOutput(ui={"images": []})

        resolved_prompt, resolved_extra_pnginfo = resolve_hidden_metadata(cls, prompt, extra_pnginfo)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        results: list[dict[str, str]] = []
        for batch_number, image in enumerate(images):
            image_data = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(image_data, 0, 255).astype(np.uint8))
            metadata = build_save_image_advanced_metadata(
                cls=cls,
                image=image,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                denoise=denoise,
                checkpoint=checkpoint,
                original_width=original_width,
                original_height=original_height,
                embed_workflow=embed_workflow,
                basic_workflow=basic_workflow,
                prompt=resolved_prompt,
                extra_pnginfo=resolved_extra_pnginfo,
            )

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            pil_image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=IMAGE_COMPRESS_LEVEL)
            results.append({"filename": file, "subfolder": subfolder, "type": IMAGE_OUTPUT_TYPE})
            counter += 1

        return IO.NodeOutput(ui={"images": results})


class SaveImageBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save Image (Bundle)",
            category=CATEGORY,
            description="Saves the image from a bundle to your ComfyUI output directory.",
            search_aliases=["save image bundle", "bundle save image", "save image"],
            inputs=[
                BundleType.Input("bundle"),
                IO.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            essentials_category="Basics",
        )

    @classmethod
    def execute(cls, bundle, filename_prefix="ComfyUI") -> IO.NodeOutput:
        return output_image_bundle(bundle, mode="save", filename_prefix=filename_prefix, cls=cls)


class PreviewImageBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Preview Image (Bundle)",
            category=CATEGORY,
            description="Previews the image from a bundle.",
            search_aliases=["preview image bundle", "bundle preview image", "preview image"],
            inputs=[
                BundleType.Input("bundle"),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            essentials_category="Basics",
        )

    @classmethod
    def execute(cls, bundle) -> IO.NodeOutput:
        return output_image_bundle(bundle, mode="preview_only", filename_prefix="ComfyUI", cls=cls)


class ImageOutput(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Image Output",
            category=CATEGORY,
            description="Outputs an image with options to preview or save.",
            search_aliases=["image output", "output image", "preview image", "save image"],
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option("preview_only", []),
                        IO.DynamicCombo.Option("save", [IO.String.Input("filename_prefix", default="ComfyUI")]),
                    ],
                ),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            essentials_category="Basics",
        )

    @classmethod
    def execute(cls, image, mode) -> IO.NodeOutput:
        filename_prefix = mode["filename_prefix"] if mode["mode"] == "save" else "ComfyUI"
        return output_image(image, mode=mode["mode"], filename_prefix=filename_prefix, cls=cls)


class ImageOutputBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Image Output (Bundle)",
            category=CATEGORY,
            description="Outputs the image from a bundle with options to preview or save.",
            search_aliases=["image output bundle", "output image bundle", "preview image bundle", "save image bundle"],
            inputs=[
                BundleType.Input("bundle"),
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option("preview_only", []),
                        IO.DynamicCombo.Option("save", [IO.String.Input("filename_prefix", default="ComfyUI")]),
                    ],
                ),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            essentials_category="Basics",
        )

    @classmethod
    def execute(cls, bundle, mode) -> IO.NodeOutput:
        filename_prefix = mode["filename_prefix"] if mode["mode"] == "save" else "ComfyUI"
        return output_image_bundle(bundle, mode=mode["mode"], filename_prefix=filename_prefix, cls=cls)


class MatchColor(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Match Color",
            category=CATEGORY,
            description="Matches the colors of a target image to a reference image using the selected transfer method.",
            search_aliases=["color match", "color transfer", "match color"],
            inputs=[
                IO.Image.Input("image_ref"),
                IO.Image.Input("image_target"),
                IO.Combo.Input("method", options=MATCH_COLOR_METHODS),
            ],
            outputs=[IO.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> IO.NodeOutput:
        return IO.NodeOutput(match_color(image_ref, image_target, method))


class MatchColorBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Match Color (Bundle)",
            category=CATEGORY,
            description="Matches the target bundle image to the reference bundle image and returns the updated target bundle.",
            search_aliases=["match color bundle", "bundle color match", "color transfer bundle"],
            inputs=[
                BundleType.Input("bundle_ref"),
                BundleType.Input("bundle_target"),
                IO.Combo.Input("method", options=MATCH_COLOR_METHODS),
            ],
            outputs=[
                BundleType.Output("BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundle_ref, bundle_target, method) -> IO.NodeOutput:
        reference_bundle = normalize_bundle(bundle_ref)
        target_bundle = normalize_bundle(bundle_target)

        reference_image = reference_bundle.get("image")
        target_image = target_bundle.get("image")
        if reference_image is None:
            raise ValueError("Match Color (Bundle) requires the reference bundle to include an image.")
        if target_image is None:
            raise ValueError("Match Color (Bundle) requires the target bundle to include an image.")

        matched_image = match_color(reference_image, target_image, method)
        return IO.NodeOutput(update_bundle(target_bundle, image=matched_image))


IMAGE_NODES: list[type[IO.ComfyNode]] = [SaveImageBundle, PreviewImageBundle, ImageOutput, ImageOutputBundle, MatchColor, MatchColorBundle]
