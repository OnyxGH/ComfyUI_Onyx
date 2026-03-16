import math

from comfy_api.latest import IO

from ..lib.nodes import get_category, get_node_id

CATEGORY = get_category("utils")


class Concatenate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = IO.Autogrow.TemplatePrefix("value", prefix="value_", min=2, max=50)

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
            node_id=get_node_id(cls.__name__),
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


UTIL_NODES: list[type[IO.ComfyNode]] = [Concatenate, CalculateTileSize]
