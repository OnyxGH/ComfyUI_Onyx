import folder_paths
import comfy.sd

from comfy_api.latest import IO

from ..lib.bundle import create_bundle
from ..lib.io import ComboTypeInput
from ..lib.nodes import get_category, get_node_id

CATEGORY = get_category("loaders")


class CheckpointLoaderBundle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Load Checkpoint (Bundle)",
            category=CATEGORY,
            description="Loads a diffusion model checkpoint and returns a bundle.",
            search_aliases=["load checkpoint", "checkpoint loader", "model loader"],
            inputs=[
                ComboTypeInput(
                    lambda: folder_paths.get_filename_list("checkpoints"),
                    "ckpt_name",
                    display_name="ckpt_name",
                    tooltip="The name of the checkpoint (model) to load.",
                ),
            ],
            outputs=[
                IO.Custom("BUNDLE").Output("BUNDLE", display_name="BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name) -> IO.NodeOutput:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[:3]
        bundle = create_bundle(ckpt_name=ckpt_name, model=model, clip=clip, vae=vae)
        return IO.NodeOutput(bundle)


LOADER_NODES: list[type[IO.ComfyNode]] = [CheckpointLoaderBundle]
