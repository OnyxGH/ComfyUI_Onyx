from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import folder_paths

import comfy.samplers

from comfy_api.latest import IO

from .nodes import get_category, get_node_id
from .io import ComboTypeOutput

BUNDLE_TYPE = IO.Custom("BUNDLE")
CATEGORY = get_category("bundle")


InputFactory = Callable[[], IO.Input]
OutputFactory = Callable[[], IO.Output]


@dataclass(frozen=True)
class BundleField:
    key: str
    output_id: str
    input_factory: InputFactory
    output_factory: OutputFactory


@dataclass(frozen=True)
class BundleNodeSpec:
    class_name: str
    display_name: str
    description: str
    field_keys: tuple[str, ...]
    search_aliases: tuple[str, ...] = field(default_factory=tuple)


def _get_model_list(category: str) -> list[str]:
    try:
        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _get_sampler_names() -> list[str]:
    try:
        return list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        return []


def _get_scheduler_names() -> list[str]:
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return []


def _bundle_input() -> IO.Input:
    return BUNDLE_TYPE.Input("bundle", optional=True)


def _bundle_output() -> IO.Output:
    return BUNDLE_TYPE.Output("BUNDLE", display_name="BUNDLE")


def _typed_field(
    key: str,
    output_id: str,
    io_type: Any,
    *,
    input_kwargs: dict[str, Any] | None = None,
    output_kwargs: dict[str, Any] | None = None,
) -> BundleField:
    resolved_input_kwargs = input_kwargs or {}
    resolved_output_kwargs = output_kwargs or {}
    return BundleField(
        key=key,
        output_id=output_id,
        input_factory=lambda: io_type.Input(
            key, optional=True, **resolved_input_kwargs
        ),
        output_factory=lambda: io_type.Output(
            output_id, display_name=output_id, **resolved_output_kwargs
        ),
    )


def _combo_field(
    key: str,
    output_id: str,
    options_provider: Callable[[], list[str]],
    *,
    force_input: bool = False,
) -> BundleField:
    return BundleField(
        key=key,
        output_id=output_id,
        input_factory=lambda: IO.Combo.Input(
            key,
            options=options_provider(),
            optional=True,
            extra_dict={"forceInput": True} if force_input else None,
        ),
        output_factory=lambda: ComboTypeOutput(
            options_provider, output_id, display_name=output_id
        ),
    )


FIELD_DEFINITIONS: dict[str, BundleField] = {
    "model": _typed_field("model", "MODEL", IO.Model),
    "model_high": _typed_field("model_high", "MODEL_HIGH", IO.Model),
    "model_low": _typed_field("model_low", "MODEL_LOW", IO.Model),
    "clip": _typed_field("clip", "CLIP", IO.Clip),
    "vae": _typed_field("vae", "VAE", IO.Vae),
    "positive": _typed_field("positive", "POSITIVE", IO.Conditioning),
    "negative": _typed_field("negative", "NEGATIVE", IO.Conditioning),
    "latent": _typed_field("latent", "LATENT", IO.Latent),
    "image": _typed_field("image", "IMAGE", IO.Image),
    "mask": _typed_field("mask", "MASK", IO.Mask),
    "audio": _typed_field("audio", "AUDIO", IO.Audio),
    "video": _typed_field("video", "VIDEO", IO.Video),
    "seed": _typed_field("seed", "SEED", IO.Int, input_kwargs={"force_input": True}),
    "steps": _typed_field("steps", "STEPS", IO.Int, input_kwargs={"force_input": True}),
    "cfg": _typed_field("cfg", "CFG", IO.Float, input_kwargs={"force_input": True}),
    "denoise": _typed_field(
        "denoise", "DENOISE", IO.Float, input_kwargs={"force_input": True}
    ),
    "start_at_step": _typed_field(
        "start_at_step", "START_AT_STEP", IO.Int, input_kwargs={"force_input": True}
    ),
    "end_at_step": _typed_field(
        "end_at_step", "END_AT_STEP", IO.Int, input_kwargs={"force_input": True}
    ),
    "width": _typed_field("width", "WIDTH", IO.Int, input_kwargs={"force_input": True}),
    "height": _typed_field(
        "height", "HEIGHT", IO.Int, input_kwargs={"force_input": True}
    ),
    "batch_size": _typed_field(
        "batch_size", "BATCH_SIZE", IO.Int, input_kwargs={"force_input": True}
    ),
    "scale_factor": _typed_field(
        "scale_factor", "SCALE_FACTOR", IO.Float, input_kwargs={"force_input": True}
    ),
    "positive_prompt": _typed_field(
        "positive_prompt",
        "POSITIVE_PROMPT",
        IO.String,
        input_kwargs={"multiline": True},
    ),
    "negative_prompt": _typed_field(
        "negative_prompt",
        "NEGATIVE_PROMPT",
        IO.String,
        input_kwargs={"multiline": True},
    ),
    "add_noise": _typed_field("add_noise", "ADD_NOISE", IO.Boolean),
    "return_with_leftover_noise": _typed_field(
        "return_with_leftover_noise", "RETURN_WITH_LEFTOVER_NOISE", IO.Boolean
    ),
    "noise": _typed_field("noise", "NOISE", IO.Noise),
    "guider": _typed_field("guider", "GUIDER", IO.Guider),
    "sampler": _typed_field("sampler", "SAMPLER", IO.Sampler),
    "sigmas": _typed_field("sigmas", "SIGMAS", IO.Sigmas),
    "ckpt_name": _combo_field(
        "ckpt_name",
        "CKPT_NAME",
        lambda: _get_model_list("checkpoints"),
        force_input=True,
    ),
    "sampler_name": _combo_field("sampler_name", "SAMPLER_NAME", _get_sampler_names),
    "scheduler": _combo_field("scheduler", "SCHEDULER", _get_scheduler_names),
    "upscale_model": _combo_field(
        "upscale_model", "UPSCALE_MODEL", lambda: _get_model_list("upscale_models")
    ),
    "any_a": _typed_field("any_a", "ANY_A", IO.AnyType),
    "any_b": _typed_field("any_b", "ANY_B", IO.AnyType),
    "any_c": _typed_field("any_c", "ANY_C", IO.AnyType),
    "any_d": _typed_field("any_d", "ANY_D", IO.AnyType),
    "any_e": _typed_field("any_e", "ANY_E", IO.AnyType),
    "segs": _typed_field("segs", "SEGS", IO.Custom("SEGS")),
    "batch_segs": _typed_field("batch_segs", "BATCH_SEGS", IO.Custom("BATCH_SEGS")),
    "sam3_model": _typed_field("sam3_model", "SAM3_MODEL", IO.Custom("SAM3_MODEL")),
    "bbox_detector": _typed_field(
        "bbox_detector", "BBOX_DETECTOR", IO.Custom("BBOX_DETECTOR")
    ),
    "segm_detector": _typed_field(
        "segm_detector", "SEGM_DETECTOR", IO.Custom("SEGM_DETECTOR")
    ),
}


BUNDLE_NODE_SPECS: tuple[BundleNodeSpec, ...] = (
    BundleNodeSpec(
        class_name="Bundle",
        display_name="Bundle",
        description="Builds or updates a core generation bundle with checkpoint, model, conditioning, and result slots.",
        field_keys=(
            "ckpt_name",
            "model",
            "clip",
            "vae",
            "positive",
            "negative",
            "seed",
            "latent",
            "image",
        ),
        search_aliases=("basic bundle", "generation bundle"),
    ),
    BundleNodeSpec(
        class_name="BundleWAN",
        display_name="Bundle (WAN)",
        description="Builds or updates a WAN-focused bundle with dual-model slots and media outputs.",
        field_keys=(
            "model_high",
            "model_low",
            "clip",
            "vae",
            "positive",
            "negative",
            "seed",
            "latent",
            "image",
            "audio",
            "video",
        ),
        search_aliases=("wan bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleSampling",
        display_name="Bundle (Sampling)",
        description="Builds or updates sampling settings on a bundle.",
        field_keys=("steps", "cfg", "sampler_name", "scheduler", "denoise"),
        search_aliases=("sampling bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleSamplingAdvanced",
        display_name="Bundle (Advanced Sampling)",
        description="Builds or updates advanced sampling controls on a bundle.",
        field_keys=(
            "add_noise",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "start_at_step",
            "end_at_step",
            "return_with_leftover_noise",
        ),
        search_aliases=("advanced sampling bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleSamplingCustom",
        display_name="Bundle (Custom Sampling)",
        description="Builds or updates custom sampling components on a bundle.",
        field_keys=("noise", "guider", "sampler", "sigmas"),
        search_aliases=("custom sampling bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleOptions",
        display_name="Bundle (Options)",
        description="Builds or updates generation option fields on a bundle.",
        field_keys=(
            "width",
            "height",
            "batch_size",
            "upscale_model",
            "scale_factor",
            "positive_prompt",
            "negative_prompt",
        ),
        search_aliases=("options bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleResults",
        display_name="Bundle (Results)",
        description="Builds or updates result slots on a bundle.",
        field_keys=("image", "latent", "mask", "audio", "video"),
        search_aliases=("results bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleDetailing",
        display_name="Bundle (Detailing)",
        description="Builds or updates detailing resources on a bundle.",
        field_keys=(
            "sam3_model",
            "bbox_detector",
            "segm_detector",
            "segs",
            "batch_segs",
        ),
        search_aliases=("detailing bundle",),
    ),
    BundleNodeSpec(
        class_name="BundleAny",
        display_name="Bundle (Any)",
        description="Builds or updates arbitrary payload slots on a bundle.",
        field_keys=("any_a", "any_b", "any_c", "any_d", "any_e"),
        search_aliases=("any bundle",),
    ),
)


def _normalize_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, tuple) and bundle:
        bundle = bundle[0]
    if isinstance(bundle, dict):
        return dict(bundle)
    return {}


def _merge_bundle(base_bundle: Any, values: dict[str, Any]) -> dict[str, Any]:
    merged = _normalize_bundle(base_bundle)
    for key, value in values.items():
        if value is not None:
            merged[key] = value
    return merged


def create_bundle(**values: Any) -> dict[str, Any]:
    return _merge_bundle(None, values)


def update_bundle(bundle: Any, **values: Any) -> dict[str, Any]:
    return _merge_bundle(bundle, values)


def _build_inputs(spec: BundleNodeSpec) -> list[IO.Input]:
    return [
        _bundle_input(),
        *[FIELD_DEFINITIONS[key].input_factory() for key in spec.field_keys],
    ]


def _build_outputs(spec: BundleNodeSpec) -> list[IO.Output]:
    return [
        _bundle_output(),
        *[FIELD_DEFINITIONS[key].output_factory() for key in spec.field_keys],
    ]


def _build_output_values(
    spec: BundleNodeSpec, bundle: dict[str, Any]
) -> tuple[Any, ...]:
    return (bundle, *[bundle.get(key) for key in spec.field_keys])


def create_bundle_node(
    spec: BundleNodeSpec, category: str = CATEGORY
) -> type[IO.ComfyNode]:
    schema = IO.Schema(
        node_id=get_node_id(spec.class_name),
        display_name=spec.display_name,
        category=category,
        description=spec.description,
        search_aliases=list(spec.search_aliases),
        inputs=_build_inputs(spec),
        outputs=_build_outputs(spec),
    )

    def define_schema(cls) -> IO.Schema:
        return schema

    def execute(cls, bundle=None, **kwargs) -> IO.NodeOutput:
        merged_bundle = _merge_bundle(bundle, kwargs)
        return IO.NodeOutput(*_build_output_values(spec, merged_bundle))

    return type(
        spec.class_name,
        (IO.ComfyNode,),
        {
            "define_schema": classmethod(define_schema),
            "execute": classmethod(execute),
        },
    )


def build_bundle_nodes(category: str = CATEGORY) -> list[type[IO.ComfyNode]]:
    return [create_bundle_node(spec, category=category) for spec in BUNDLE_NODE_SPECS]