import folder_paths

import comfy.samplers

from comfy_api.latest import IO

from ..helpers.io import ComboTypeInput, ComboTypeOutput
from ..helpers.nodes import get_category, get_node_id

CATEGORY = get_category("widgets")


class CheckpointName(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Checkpoint Name",
            category=CATEGORY,
            description="Provides the name of a checkpoint (model) from the checkpoints folder.",
            search_aliases=["checkpoint name", "model name", "ckpt name"],
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


class Seed(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Seed",
            category=CATEGORY,
            description="Provides a seed value for random number generation.",
            search_aliases=["seed", "random seed"],
            inputs=[
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[IO.Int.Output("SEED", display_name="SEED")],
        )

    @classmethod
    def execute(cls, seed) -> IO.NodeOutput:
        return IO.NodeOutput(seed)


# region KSampler Widgets


class Steps(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Steps",
            category=CATEGORY,
            description="Provides a steps value for a KSampler.",
            search_aliases=["steps", "num steps", "number of steps"],
            inputs=[
                IO.Int.Input("steps", default=20, min=1, max=10000),
            ],
            outputs=[IO.Int.Output("STEPS", display_name="STEPS")],
        )

    @classmethod
    def execute(cls, steps) -> IO.NodeOutput:
        return IO.NodeOutput(steps)


class Cfg(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="CFG",
            category=CATEGORY,
            description="Provides a CFG scale value for a KSampler.",
            search_aliases=["cfg", "cfg scale", "classifier-free guidance scale"],
            inputs=[
                IO.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
            ],
            outputs=[IO.Float.Output("CFG", display_name="CFG")],
        )

    @classmethod
    def execute(cls, cfg) -> IO.NodeOutput:
        return IO.NodeOutput(cfg)


class SamplerName(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Sampler Name",
            category=CATEGORY,
            description="Provides the name of a sampler for use in a KSampler.",
            search_aliases=["sampler name", "sampler", "k sampler"],
            inputs=[
                ComboTypeInput(
                    lambda: comfy.samplers.KSampler.SAMPLERS,
                    "sampler_name",
                    display_name="sampler_name",
                    tooltip="The name of the sampler to use in a KSampler.",
                ),
            ],
            outputs=[
                ComboTypeOutput(
                    lambda: comfy.samplers.KSampler.SAMPLERS,
                    "SAMPLER_NAME",
                    display_name="SAMPLER_NAME",
                ),
            ],
        )

    @classmethod
    def execute(cls, sampler_name) -> IO.NodeOutput:
        return IO.NodeOutput(sampler_name)


class Scheduler(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Scheduler",
            category=CATEGORY,
            description="Provides the name of a scheduler for use in a KSampler.",
            search_aliases=["scheduler", "k sampler scheduler"],
            inputs=[
                ComboTypeInput(
                    lambda: comfy.samplers.KSampler.SCHEDULERS,
                    "scheduler_name",
                    display_name="scheduler_name",
                    tooltip="The name of the scheduler to use in a KSampler.",
                ),
            ],
            outputs=[
                ComboTypeOutput(
                    lambda: comfy.samplers.KSampler.SCHEDULERS,
                    "SCHEDULER_NAME",
                    display_name="SCHEDULER_NAME",
                ),
            ],
        )

    @classmethod
    def execute(cls, scheduler_name) -> IO.NodeOutput:
        return IO.NodeOutput(scheduler_name)


class Denoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Denoise",
            category=CATEGORY,
            description="Provides a denoise value for a KSampler.",
            search_aliases=["denoise", "denoise strength", "denoising strength"],
            inputs=[
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, round=0.01),
            ],
            outputs=[IO.Float.Output("DENOISE", display_name="DENOISE")],
        )

    @classmethod
    def execute(cls, denoise) -> IO.NodeOutput:
        return IO.NodeOutput(denoise)


# endregion


WIDGET_NODES: list[type[IO.ComfyNode]] = [
    CheckpointName,
    Seed,
    Steps,
    Cfg,
    SamplerName,
    Scheduler,
    Denoise,
]
