from typing import Any

from comfy_api.latest import IO
from comfy_extras.nodes_model_merging import (
    ModelMergeSimple,
    CLIPMergeSimple,
    # ModelSubtract,
    # ModelAdd,
    # CLIPSubtract,
    # CLIPAdd,
    # ModelMergeBlocks,
    CheckpointSave,
    CLIPSave,
    VAESave,
    ModelSave,
)

from nodes import VAELoader

from ...helpers.io import ComboTypeInput
from ...lib.bundle import create_bundle, normalize_bundle, update_bundle
from ...helpers.model_merge import merge_checkpoints
from ...helpers.nodes import get_category, get_node_id

CATEGORY = get_category("model/merging")


def _inherit_input() -> IO.Input:
    return IO.Combo.Input(
        "inherit_from",
        options=["bundle_a", "bundle_b"],
        default="bundle_a",
        tooltip="Determines which input bundle provides the non-merged fields in the output bundle.",
    )


def _merge_base_bundle(bundle_a, bundle_b, inherit_from) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    left_bundle = normalize_bundle(bundle_a)
    right_bundle = normalize_bundle(bundle_b)
    if inherit_from == "bundle_a":
        merged_bundle = update_bundle(left_bundle)
    elif inherit_from == "bundle_b":
        merged_bundle = update_bundle(right_bundle)
    else:
        raise ValueError(f"Unsupported inherit_from value: {inherit_from}")
    merged_bundle.pop("ckpt_name", None)
    return left_bundle, right_bundle, merged_bundle


def _require_bundle_value(bundle: Any, key: str, label: str) -> Any:
    value = normalize_bundle(bundle).get(key)
    if value is None:
        raise ValueError(f"{label} requires the input bundle to include {key}.")
    return value


# region Bundle Variants of ComfyUI Model Merging Nodes


class ModelMergeBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Model Merge (Bundle)",
            category=CATEGORY,
            description="Merges the model from two bundles together using the specified ratio and returns the merged bundle.",
            search_aliases=["model merge bundle", "merge model bundles", "model merge", "merge models"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle_a"),
                IO.Custom("BUNDLE").Input("bundle_b"),
                _inherit_input(),
                IO.Float.Input("ratio", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Custom("BUNDLE").Output("BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundle_a, bundle_b, inherit_from, ratio) -> IO.NodeOutput:
        left_bundle, right_bundle, merged_bundle = _merge_base_bundle(bundle_a, bundle_b, inherit_from)

        model_a = left_bundle.get("model")
        model_b = right_bundle.get("model")
        if model_a is None or model_b is None:
            raise ValueError("Model Merge (Bundle) requires both input bundles to include a model.")

        merged_bundle["model"] = ModelMergeSimple().merge(model_a, model_b, ratio)[0]

        return IO.NodeOutput(create_bundle(**merged_bundle))


class CLIPMergeBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="CLIP Merge (Bundle)",
            category=CATEGORY,
            description="Merges the CLIP from two bundles together using the specified ratio and returns the merged bundle.",
            search_aliases=["clip merge bundle", "merge clip bundles", "clip merge", "merge clips"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle_a"),
                IO.Custom("BUNDLE").Input("bundle_b"),
                _inherit_input(),
                IO.Float.Input("ratio", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Custom("BUNDLE").Output("BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundle_a, bundle_b, inherit_from, ratio) -> IO.NodeOutput:
        left_bundle, right_bundle, merged_bundle = _merge_base_bundle(bundle_a, bundle_b, inherit_from)

        clip_a = left_bundle.get("clip")
        clip_b = right_bundle.get("clip")
        if clip_a is None or clip_b is None:
            raise ValueError("CLIP Merge (Bundle) requires both input bundles to include a clip.")

        merged_bundle["clip"] = CLIPMergeSimple().merge(clip_a, clip_b, ratio)[0]
        return IO.NodeOutput(create_bundle(**merged_bundle))


class CheckpointSaveBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save Checkpoint (Bundle)",
            category=CATEGORY,
            description="Saves the model, CLIP, and VAE from a bundle as a checkpoint.",
            search_aliases=["checkpoint save bundle", "save checkpoint bundle", "export checkpoint bundle"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle"),
                IO.String.Input("filename_prefix", default="checkpoints/ComfyUI"),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, bundle, filename_prefix, prompt=None, extra_pnginfo=None) -> IO.NodeOutput:
        normalized_bundle = normalize_bundle(bundle)
        model = _require_bundle_value(normalized_bundle, "model", "Checkpoint Save (Bundle)")
        clip = _require_bundle_value(normalized_bundle, "clip", "Checkpoint Save (Bundle)")
        vae = _require_bundle_value(normalized_bundle, "vae", "Checkpoint Save (Bundle)")
        CheckpointSave().save(model, clip, vae, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return IO.NodeOutput()


class CLIPSaveBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save CLIP (Bundle)",
            category=CATEGORY,
            description="Saves the CLIP from a bundle.",
            search_aliases=["clip save bundle", "save clip bundle", "export clip bundle"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle"),
                IO.String.Input("filename_prefix", default="clip/ComfyUI"),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, bundle, filename_prefix, prompt=None, extra_pnginfo=None) -> IO.NodeOutput:
        clip = _require_bundle_value(bundle, "clip", "CLIP Save (Bundle)")
        CLIPSave().save(clip, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return IO.NodeOutput()


class VAESaveBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save VAE (Bundle)",
            category=CATEGORY,
            description="Saves the VAE from a bundle.",
            search_aliases=["vae save bundle", "save vae bundle", "export vae bundle"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle"),
                IO.String.Input("filename_prefix", default="vae/ComfyUI_vae"),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, bundle, filename_prefix, prompt=None, extra_pnginfo=None) -> IO.NodeOutput:
        vae = _require_bundle_value(bundle, "vae", "VAE Save (Bundle)")
        VAESave().save(vae, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return IO.NodeOutput()


class ModelSaveBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Save Model (Bundle)",
            category=CATEGORY,
            description="Saves the model from a bundle.",
            search_aliases=["model save bundle", "save model bundle", "export model bundle"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle"),
                IO.String.Input("filename_prefix", default="diffusion_models/ComfyUI"),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, bundle, filename_prefix, prompt=None, extra_pnginfo=None) -> IO.NodeOutput:
        model = _require_bundle_value(bundle, "model", "Model Save (Bundle)")
        ModelSave().save(model, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return IO.NodeOutput()


# endregion


class CheckpointMerge(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Checkpoint Merge",
            category=CATEGORY,
            description="Merges two models and two CLIPs using separate ratios and returns the merged model and CLIP.",
            search_aliases=["checkpoint merge", "merge checkpoint", "model clip merge", "merge model and clip"],
            inputs=[
                IO.Model.Input("model_a"),
                IO.Model.Input("model_b"),
                IO.Clip.Input("clip_a"),
                IO.Clip.Input("clip_b"),
                IO.Float.Input("model_ratio", default=0.5, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("clip_ratio", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Model.Output("MODEL"),
                IO.Clip.Output("CLIP"),
            ],
        )

    @classmethod
    def execute(cls, model_a, model_b, clip_a, clip_b, model_ratio, clip_ratio) -> IO.NodeOutput:
        merged_model, merged_clip = merge_checkpoints(model_a, model_b, clip_a, clip_b, model_ratio, clip_ratio)
        return IO.NodeOutput(merged_model, merged_clip)


class CheckpointMergeBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Checkpoint Merge (Bundle)",
            category=CATEGORY,
            description="Merges the models and CLIPs from two bundles together using the specified ratios and returns the merged bundle.",
            search_aliases=["checkpoint merge bundle", "merge checkpoint bundles", "model clip merge bundle", "merge model and clip bundle"],
            inputs=[
                IO.Custom("BUNDLE").Input("bundle_a"),
                IO.Custom("BUNDLE").Input("bundle_b"),
                _inherit_input(),
                IO.Float.Input("model_ratio", default=0.5, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("clip_ratio", default=0.5, min=0.0, max=1.0, step=0.01),
                IO.DynamicCombo.Input(
                    "vae_select",
                    tooltip="Determines how to handle the VAE in the output bundle since the merging process only merges the model and CLIP. Choose 'from_bundle' to take the VAE from the same bundle as the non-merged fields, 'load_vae' to load a specific VAE and include it in the output bundle, or 'none' to exclude the VAE from the output bundle.",
                    options=[
                        IO.DynamicCombo.Option("from_bundle", []),
                        IO.DynamicCombo.Option(
                            "load_vae",
                            [
                                ComboTypeInput(
                                    lambda: VAELoader.vae_list({"video_taes": []}),
                                    "vae_name",
                                    tooltip="The name of the VAE to load and include in the output bundle.",
                                )
                            ],
                        ),
                        IO.DynamicCombo.Option("none", []),
                    ],
                ),
            ],
            outputs=[
                IO.Custom("BUNDLE").Output("BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundle_a, bundle_b, inherit_from, model_ratio, clip_ratio, vae_select) -> IO.NodeOutput:
        left_bundle, right_bundle, merged_bundle = _merge_base_bundle(bundle_a, bundle_b, inherit_from)

        model_a = left_bundle.get("model")
        model_b = right_bundle.get("model")
        clip_a = left_bundle.get("clip")
        clip_b = right_bundle.get("clip")
        if model_a is None or model_b is None or clip_a is None or clip_b is None:
            raise ValueError("Checkpoint Merge (Bundle) requires both input bundles to include a model and a clip.")

        merged_model, merged_clip = merge_checkpoints(model_a, model_b, clip_a, clip_b, model_ratio, clip_ratio)
        merged_bundle["model"] = merged_model
        merged_bundle["clip"] = merged_clip

        if vae_select == "from_bundle":
            vae_a = left_bundle.get("vae")
            vae_b = right_bundle.get("vae")
            if vae_a is None or vae_b is None:
                raise ValueError("Checkpoint Merge (Bundle) with 'from_bundle' VAE selection requires both input bundles to include a VAE.")
            merged_bundle["vae"] = vae_a if inherit_from == "bundle_a" else vae_b
        elif vae_select == "load_vae":
            vae_name = cls.get_input_value("vae_name")
            if not vae_name:
                raise ValueError("Checkpoint Merge (Bundle) with 'load_vae' VAE selection requires a VAE name to be selected.")
            merged_bundle["vae"] = VAELoader.load_vae(vae_name)[0]
        elif vae_select == "none":
            merged_bundle.pop("vae", None)

        return IO.NodeOutput(create_bundle(**merged_bundle))


MODEL_MERGE_NODES: list[type[IO.ComfyNode]] = [
    ModelMergeBundle,
    CLIPMergeBundle,
    CheckpointSaveBundle,
    CLIPSaveBundle,
    VAESaveBundle,
    ModelSaveBundle,
    CheckpointMerge,
    CheckpointMergeBundle,
]
