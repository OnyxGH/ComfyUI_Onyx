import nodes

from comfy_api.latest import IO

from ...lib.nodes import get_category, get_node_id
from ...lib.tiled_diffusion import (
    MultiDiffusion as MultiDiffusionImpl,
    MixtureOfDiffusers as MixtureOfDiffusersImpl,
    SpotDiffusion as SpotDiffusionImpl,
)

CATEGORY = get_category("model/patches")

_DEFAULT_COMPRESSION = 8
_TILED_DIFFUSION_METHODS = {
    "MultiDiffusion": MultiDiffusionImpl,
    "Mixture of Diffusers": MixtureOfDiffusersImpl,
    "SpotDiffusion": SpotDiffusionImpl,
}


class TiledDiffusion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls.__name__),
            category=CATEGORY,
            display_name="Tiled Diffusion",
            description="Applies a tiled diffusion patch to a model using the selected method and tile settings.",
            search_aliases=["tiled diffusion", "multi diffusion", "mixture of diffusers", "spot diffusion"],
            inputs=[
                IO.Model.Input("model"),
                IO.Combo.Input(
                    "method",
                    options=list(_TILED_DIFFUSION_METHODS.keys()),
                    default="Mixture of Diffusers",
                ),
                IO.Int.Input("tile_width", default=96 * _DEFAULT_COMPRESSION, min=16, max=nodes.MAX_RESOLUTION, step=16),
                IO.Int.Input("tile_height", default=96 * _DEFAULT_COMPRESSION, min=16, max=nodes.MAX_RESOLUTION, step=16),
                IO.Int.Input(
                    "tile_overlap",
                    default=8 * _DEFAULT_COMPRESSION,
                    min=0,
                    max=256 * _DEFAULT_COMPRESSION,
                    step=4 * _DEFAULT_COMPRESSION,
                ),
                IO.Int.Input("tile_batch_size", default=4, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Model.Output("MODEL")],
        )

    @classmethod
    def _apply_tiled_diffusion_settings(cls, impl, model, tile_width, tile_height, tile_overlap, tile_batch_size):
        compression = 4 if "CASCADE" in str(model.model.model_type) else _DEFAULT_COMPRESSION
        impl.tile_width = tile_width // compression
        impl.tile_height = tile_height // compression
        impl.tile_overlap = tile_overlap // compression
        impl.tile_batch_size = tile_batch_size
        impl.compression = compression
        impl.width = tile_width
        impl.height = tile_height
        impl.overlap = tile_overlap

    @classmethod
    def execute(cls, model, method, tile_width, tile_height, tile_overlap, tile_batch_size) -> IO.NodeOutput:
        impl_class = _TILED_DIFFUSION_METHODS.get(method)
        if impl_class is None:
            raise ValueError(f"Unsupported tiled diffusion method: {method}")

        impl = impl_class()
        cls._apply_tiled_diffusion_settings(impl, model, tile_width, tile_height, tile_overlap, tile_batch_size)

        model_out = model.clone()
        model_out.set_model_unet_function_wrapper(impl)
        model_out.model_options["tiled_diffusion"] = True
        return IO.NodeOutput(model_out)


class SpotDiffusionParams(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls.__name__),
            category=CATEGORY,
            display_name="Spot Diffusion Params",
            description="Applies Spot Diffusion shift settings to a model for tiled diffusion sampling.",
            search_aliases=["spot diffusion", "spot diffusion params", "shift method", "tiled diffusion seed"],
            inputs=[
                IO.Model.Input("model"),
                IO.Combo.Input(
                    "shift_method",
                    options=["random", "sorted", "fibonacci"],
                    default="random",
                    tooltip="Samples a shift size over a uniform distribution to shift tiles.",
                ),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[IO.Model.Output("MODEL")],
        )

    @classmethod
    def execute(cls, model, shift_method, seed) -> IO.NodeOutput:
        model_out = model.clone()
        model_out.model_options["tiled_diffusion_seed"] = seed
        model_out.model_options["tiled_diffusion_shift_method"] = shift_method
        return IO.NodeOutput(model_out)


MODEL_PATCH_NODES: list[type[IO.ComfyNode]] = [TiledDiffusion, SpotDiffusionParams]
