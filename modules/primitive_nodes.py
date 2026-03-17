import sys

from comfy_api.latest import IO

from ..helpers.nodes import get_category, get_node_id

CATEGORY = get_category("widgets/primitives")


class Float(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Float",
            category=CATEGORY,
            description="Provides a float value.",
            search_aliases=["float", "decimal", "number"],
            inputs=[
                IO.Float.Input("value", display_name="value", min=-sys.maxsize, max=sys.maxsize, step=0.01),
            ],
            outputs=[
                IO.Float.Output("float", display_name="FLOAT"),
            ],
        )

    @classmethod
    def execute(cls, value) -> IO.NodeOutput:
        return IO.NodeOutput(value)


class Int(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Int",
            category=CATEGORY,
            description="Provides an integer value.",
            search_aliases=["int", "integer", "whole number"],
            inputs=[
                IO.Int.Input("value", display_name="value", min=-sys.maxsize, max=sys.maxsize),
            ],
            outputs=[
                IO.Int.Output("int", display_name="INT"),
            ],
        )

    @classmethod
    def execute(cls, value) -> IO.NodeOutput:
        return IO.NodeOutput(value)


class Ratio(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Ratio",
            category=CATEGORY,
            description="Provides a ratio value between 0.0 and 1.0.",
            search_aliases=["ratio", "percentage", "proportion"],
            inputs=[
                IO.Float.Input("ratio", display_name="ratio", min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Float.Output("RATIO", display_name="RATIO"),
            ],
        )

    @classmethod
    def execute(cls, ratio) -> IO.NodeOutput:
        return IO.NodeOutput(ratio)


PRIMITIVE_NODES: list[type[IO.ComfyNode]] = [Float, Int, Ratio]
