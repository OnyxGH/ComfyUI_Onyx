import folder_paths

from comfy_api.latest import IO

from ..lib.io import ComboTypeOutput
from ..lib.nodes import get_category, get_node_id

CATEGORY = get_category("widgets")


class CheckpointName(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Checkpoint Name",
            category=CATEGORY,
            inputs=[
                IO.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
            ],
            outputs=[
                ComboTypeOutput(
                    lambda: folder_paths.get_filename_list("checkpoints"),
                    "CKPT_NAME",
                    display_name="CKPT_NAME",
                ),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name) -> IO.NodeOutput:
        return IO.NodeOutput(ckpt_name)


WIDGET_NODES: list[type[IO.ComfyNode]] = [CheckpointName]
