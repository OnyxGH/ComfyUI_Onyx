from comfy_api.latest import IO

from ..lib.math import safe_eval
from ..lib.nodes import get_category, get_node_id
from ..lib.constants import ALPHABET_LOWER

CATEGORY = get_category("math")

_EVAL_INPUT_NAMES = list(ALPHABET_LOWER)


def _extract_variables(values: IO.Autogrow.Type) -> dict[str, float]:
    variables: dict[str, float] = {}
    for key, val in values.items():
        if isinstance(val, (int, float)):
            variables[key] = float(val)
    return variables


class Evaluate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = IO.Autogrow.TemplateNames(
            IO.MultiType.Input("val", types=[IO.Float, IO.Int], optional=True),
            names=_EVAL_INPUT_NAMES,
            min=1,
        )

        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Evaluate",
            category=CATEGORY,
            description="Evaluates a mathematical expression using the provided inputs as variables.",
            search_aliases=["evaluate", "calculate", "compute"],
            inputs=[
                IO.Autogrow.Input("values", template=autogrow_template),
                IO.String.Input("expression", display_name="expression", multiline=True),
            ],
            outputs=[
                IO.Float.Output("FLOAT"),
                IO.Int.Output("INT"),
            ],
        )

    @classmethod
    def execute(cls, values: IO.Autogrow.Type, expression: str) -> IO.NodeOutput:
        variables = _extract_variables(values)

        try:
            result = safe_eval(expression, variables)
        except Exception:
            result = 0.0

        return IO.NodeOutput(result, int(result))


MATH_NODES: list[type[IO.ComfyNode]] = [Evaluate]
