from __future__ import annotations

import numpy as np
import torch

from torch.nn import functional as F
from comfy_api.latest import IO, UI

from ..lib.color_correction import (
    adain_color_fix,
    lab_affine_color_correct_tensor,
    moment,
    paired_pixel_color_correction,
    pil2tensor,
    root_poly_color_correct_tensor,
    tensor2pil,
    wavelet_color_fix,
)
from ..lib.bundle import BundleType, normalize_bundle, update_bundle
from ..lib.nodes import get_category, get_node_id

CATEGORY = get_category("image")
_MATCH_COLOR_METHODS = [
    "wavelet",
    "adain",
    "moment",
    "poly",
    "ccm",
    "lab",
    "mkl",
    "hm",
    "reinhard",
    "mvgd",
    "hm-mvgd-hm",
    "hm-mkl-hm",
]
_COLOR_MATCHER_METHODS = {"mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"}


def _output_image(image: torch.Tensor, mode: str, filename_prefix: str, cls: type[IO.ComfyNode]) -> IO.NodeOutput:
    if mode == "preview_only":
        return IO.NodeOutput(ui=UI.PreviewImage(image, cls=cls))
    elif mode == "save":
        return IO.NodeOutput(ui=UI.ImageSaveHelper.get_save_images_ui(images=image, filename_prefix=filename_prefix, cls=cls))
    else:
        raise ValueError(f"Unsupported mode for outputting image: {mode}")


def _output_image_bundle(bundle, mode: str, filename_prefix: str, cls: type[IO.ComfyNode]) -> IO.NodeOutput:
    normalized_bundle = normalize_bundle(bundle)
    if "image" not in normalized_bundle:
        raise ValueError("The bundle must include an image for output.")
    image = normalized_bundle["image"]
    return _output_image(image, mode, filename_prefix, cls=cls)


def _resolve_reference_image(image_ref: torch.Tensor, batch_size: int, index: int) -> torch.Tensor:
    if image_ref.size(0) == 1:
        return image_ref[:1]
    if image_ref.size(0) != batch_size:
        raise ValueError("Match Color expects either a single reference image or a batch matching the target image batch size.")
    return image_ref[index : index + 1]


def _resize_like(image: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[1:3]) == spatial_shape:
        return image

    channels_first = image.permute(0, 3, 1, 2)
    resized = F.interpolate(channels_first, size=spatial_shape, mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _match_with_color_matcher(image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> torch.Tensor:
    try:
        from color_matcher import ColorMatcher  # type: ignore
    except ImportError as exc:
        raise ImportError("Match Color requires the 'color-matcher' package for this method.") from exc

    matcher = ColorMatcher()
    outputs: list[torch.Tensor] = []
    batch_size = image_target.size(0)

    for index in range(batch_size):
        target_image = image_target[index : index + 1]
        reference_image = _resolve_reference_image(image_ref, batch_size, index)
        if reference_image.shape[1:3] != target_image.shape[1:3]:
            reference_image = _resize_like(reference_image, tuple(target_image.shape[1:3]))

        result = matcher.transfer(
            src=target_image[0].detach().cpu().numpy(),
            ref=reference_image[0].detach().cpu().numpy(),
            method=method,
        )
        outputs.append(torch.from_numpy(np.asarray(result, dtype=np.float32)))

    return torch.stack(outputs, dim=0)


def _match_color(image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> torch.Tensor:
    batch_size = image_target.size(0)
    outputs: list[torch.Tensor] = []

    if method in {"wavelet", "adain"}:
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = _resolve_reference_image(image_ref, batch_size, index)
            result_pil = (
                wavelet_color_fix(tensor2pil(target_image), tensor2pil(reference_image))
                if method == "wavelet"
                else adain_color_fix(tensor2pil(target_image), tensor2pil(reference_image))
            )
            outputs.append(pil2tensor(result_pil).to(image_target.device))
        return torch.cat(outputs, dim=0)

    if method == "moment":
        for index in range(batch_size):
            outputs.append(
                moment(
                    image=image_target[index : index + 1],
                    reference=_resolve_reference_image(image_ref, batch_size, index),
                    strength=1.0,
                    adaptive_matching=True,
                    device=image_target.device,
                )
            )
        return torch.cat(outputs, dim=0)

    if method == "ccm":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = _resolve_reference_image(image_ref, batch_size, index)
            outputs.append(paired_pixel_color_correction(target_image, reference_image, degree=1))
        return torch.cat(outputs, dim=0)

    if method == "poly":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = _resolve_reference_image(image_ref, batch_size, index)
            if reference_image.shape[1:3] != target_image.shape[1:3]:
                reference_image = _resize_like(reference_image, tuple(target_image.shape[1:3]))
            outputs.append(root_poly_color_correct_tensor(reference_image, target_image, 0.1, 200_000, 1e-3))
        return torch.cat(outputs, dim=0)

    if method == "lab":
        for index in range(batch_size):
            target_image = image_target[index : index + 1]
            reference_image = _resolve_reference_image(image_ref, batch_size, index)
            if reference_image.shape[1:3] != target_image.shape[1:3]:
                reference_image = _resize_like(reference_image, tuple(target_image.shape[1:3]))
            outputs.append(lab_affine_color_correct_tensor(reference_image, target_image, 0.1, 200_000, 1e-3))
        return torch.cat(outputs, dim=0)

    if method in _COLOR_MATCHER_METHODS:
        return _match_with_color_matcher(image_ref, image_target, method).to(image_target.device)

    raise ValueError(f"Unsupported Match Color method: {method}")


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
        return _output_image_bundle(bundle, mode="save", filename_prefix=filename_prefix, cls=cls)


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
        return _output_image_bundle(bundle, mode="preview_only", filename_prefix="ComfyUI", cls=cls)


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
        return _output_image(image, mode=mode["mode"], filename_prefix=filename_prefix, cls=cls)


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
        return _output_image_bundle(bundle, mode=mode["mode"], filename_prefix=filename_prefix, cls=cls)


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
                IO.Combo.Input("method", options=_MATCH_COLOR_METHODS),
            ],
            outputs=[IO.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> IO.NodeOutput:
        return IO.NodeOutput(_match_color(image_ref, image_target, method))


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
                IO.Combo.Input("method", options=_MATCH_COLOR_METHODS),
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

        matched_image = _match_color(reference_image, target_image, method)
        return IO.NodeOutput(update_bundle(target_bundle, image=matched_image))


IMAGE_NODES: list[type[IO.ComfyNode]] = [SaveImageBundle, PreviewImageBundle, ImageOutput, ImageOutputBundle, MatchColor, MatchColorBundle]
