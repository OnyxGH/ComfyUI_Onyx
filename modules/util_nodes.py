import math
import nodes

from nodes import EmptyLatentImage

from comfy_api.latest import IO

from ..lib.nodes import get_category, get_node_id
from ..lib.bundle import BundleType, update_bundle

CATEGORY = get_category("utils")


class Concatenate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = IO.Autogrow.TemplatePrefix(IO.String.Input("value"), prefix="value_", min=2, max=50)

        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Concatenate",
            category=CATEGORY,
            description="Concatenates multiple string inputs with an optional delimiter.",
            search_aliases=["concat", "join", "combine"],
            inputs=[
                IO.Autogrow.Input("values", template=autogrow_template),
                IO.DynamicCombo.Input(
                    "delimiter_type",
                    tooltip="Choose the type of delimiter to use for concatenation.",
                    options=[
                        IO.DynamicCombo.Option("single_line", [IO.String.Input("delimiter_single", display_name="delimiter", default="")]),
                        IO.DynamicCombo.Option("multi_line", [IO.String.Input("delimiter_multi", display_name="delimiter", default="", multiline=True)]),
                    ],
                ),
            ],
            outputs=[
                IO.String.Output("STRING", display_name="STRING"),
            ],
        )

    @classmethod
    def execute(cls, values: IO.Autogrow.Type, delimiter_type) -> IO.NodeOutput:
        non_empty_values = [str(v) for v in values.values() if v is not None and str(v).strip() != ""]
        if delimiter_type["delimiter_type"] == "single_line":
            delimiter = delimiter_type["delimiter_single"]
        elif delimiter_type["delimiter_type"] == "multi_line":
            delimiter = delimiter_type["delimiter_multi"]
        else:
            delimiter = ""
        result = delimiter.join(non_empty_values)
        return IO.NodeOutput(result)


class CalculateTileSize(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Calculate Tile Size",
            category=CATEGORY,
            description="Calculates optimal tile size for image processing based on input image dimensions, tile block size, and upscale factor.",
            search_aliases=["tile size", "calculate tile", "tile calculator"],
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("tile_block", tooltip="Specify the tile block size.", default=1536, min=512, max=2048, step=256),
                IO.Float.Input("upscale_factor", tooltip="Specify the upscale factor.", default=1.0, min=0.5, max=4.0, step=0.05, round=0.01),
            ],
            outputs=[
                IO.Int.Output("TILE_WIDTH", display_name="TILE_WIDTH"),
                IO.Int.Output("TILE_HEIGHT", display_name="TILE_HEIGHT"),
            ],
        )

    @classmethod
    def execute(cls, image, tile_block: int, upscale_factor: float) -> IO.NodeOutput:
        width, height = image.shape[2], image.shape[1]

        tile_size = lambda x, p: (int(x * p) if int(x * p) < tile_block else int(int(x * p) / math.ceil(int(x * p) / tile_block)))
        upscale_by = upscale_factor if upscale_factor < 4.0 else 4.0
        tile_width = ((tile_size(width, upscale_by) + 15) // 16) * 16
        tile_height = ((tile_size(height, upscale_by) + 15) // 16) * 16

        return IO.NodeOutput(tile_width, tile_height)


class Options(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Options",
            category=CATEGORY,
            description="Provides a way to set various options for image generation, including seed, dimensions, batch size, and prompt encoding.",
            search_aliases=["options", "generation options", "image generation options"],
            inputs=[
                BundleType.Input("bundle"),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("width", default=512, min=16, max=nodes.MAX_RESOLUTION, step=8, tooltip="The width of the images in pixels."),
                IO.Int.Input("height", default=512, min=16, max=nodes.MAX_RESOLUTION, step=8, tooltip="The height of the images in pixels."),
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of images to generate in a batch."),
                IO.String.Input("positive", multiline=True, dynamic_prompts=True),
                IO.String.Input("negative", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                BundleType.Output("BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundle, seed, width, height, batch_size, positive, negative) -> IO.NodeOutput:
        normalized_bundle = bundle[0] if isinstance(bundle, tuple) and bundle else bundle

        if not isinstance(normalized_bundle, dict):
            raise ValueError("Options requires the input bundle to be a bundle dictionary.")

        if not "clip" in normalized_bundle:
            raise ValueError("Bundle must contain a CLIP model to encode prompts.")

        clip = normalized_bundle["clip"]
        tokens_pos = clip.tokenize(positive)
        tokens_neg = clip.tokenize(negative)
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

        latent = EmptyLatentImage().generate(width, height, batch_size)[0]

        updated_bundle = update_bundle(
            normalized_bundle,
            seed=seed,
            width=width,
            height=height,
            batch_size=batch_size,
            positive=cond_pos,
            negative=cond_neg,
            latent=latent,
        )
        return IO.NodeOutput(updated_bundle)


UTIL_NODES: list[type[IO.ComfyNode]] = [Concatenate, CalculateTileSize, Options]
