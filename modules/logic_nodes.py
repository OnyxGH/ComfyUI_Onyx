from comfy_api.latest import IO

from ..lib.nodes import get_category, get_node_id
from ..lib.constants import ALPHABET_LOWER

CATEGORY = get_category("logic")

_SELECT_INPUT_NAMES = list(ALPHABET_LOWER)


class Compare(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Compare",
            category=CATEGORY,
            description="Compares two inputs for the selected operation.",
            search_aliases=["compare", "equals", "greater than", "less than"],
            inputs=[
                IO.AnyType.Input("a", display_name="a"),
                IO.AnyType.Input("b", display_name="b"),
                IO.Combo.Input("operation", display_name="operation", options=["==", "!=", ">=", "<=", ">", "<"]),
            ],
            outputs=[
                IO.Boolean.Output("BOOLEAN", display_name="BOOLEAN"),
            ],
        )

    @classmethod
    def execute(cls, a, b, operation) -> IO.NodeOutput:
        result = False
        try:
            if operation == "==":
                result = a == b
            elif operation == "!=":
                result = a != b
            elif operation == ">=":
                result = a >= b
            elif operation == "<=":
                result = a <= b
            elif operation == ">":
                result = a > b
            elif operation == "<":
                result = a < b
        except Exception:
            result = False
        return IO.NodeOutput(result)


class Select(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        match_template = IO.MatchType.Template("select")
        autogrow_template = IO.Autogrow.TemplateNames(
            IO.MatchType.Input("value", template=match_template),
            names=_SELECT_INPUT_NAMES,
            min=2,
        )

        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Select",
            category=CATEGORY,
            description="Selects one of the inputs based on the selected option.",
            search_aliases=["select", "choose"],
            inputs=[
                IO.Autogrow.Input("values", template=autogrow_template),
                IO.Combo.Input("select", options=_SELECT_INPUT_NAMES, default="a"),
            ],
            outputs=[
                IO.MatchType.Output(match_template, display_name="OUTPUT"),
            ],
        )

    @classmethod
    def execute(cls, values: IO.Autogrow.Type, select: str) -> IO.NodeOutput:
        selected_value = values.get(select)
        return IO.NodeOutput(selected_value)


LOGIC_NODES: list[type[IO.ComfyNode]] = [Compare, Select]