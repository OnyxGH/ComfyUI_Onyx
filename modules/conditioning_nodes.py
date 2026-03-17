import nodes

from comfy_api.latest import IO

from ..helpers.nodes import get_category, get_node_id

CATEGORY = get_category("conditioning")


class PromptTextEncode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Prompt Text Encode",
            category=CATEGORY,
            description="Encodes positive and negative text prompts using a CLIP text encoder.",
            search_aliases=["prompt encode", "text encode", "clip encode"],
            inputs=[
                IO.Clip.Input("clip"),
                IO.String.Input("positive", multiline=True),
                IO.String.Input("negative", multiline=True),
            ],
            outputs=[
                IO.Conditioning.Output("POSITIVE", display_name="POSITIVE"),
                IO.Conditioning.Output("NEGATIVE", display_name="NEGATIVE"),
            ],
        )

    @classmethod
    def execute(cls, clip, positive, negative) -> IO.NodeOutput:
        if clip is None:
            raise RuntimeError("Clip input is invalid or not connected")
        tokens_pos = clip.tokenize(positive)
        tokens_neg = clip.tokenize(negative)
        return IO.NodeOutput(clip.encode_from_tokens_scheduled(tokens_pos), clip.encode_from_tokens_scheduled(tokens_neg))


class PromptTextEncodeSDXL(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SDXL Prompt Text Encode",
            category=CATEGORY,
            description="Encodes positive and negative text prompts using a CLIP text encoder, optimized for SDXL.",
            search_aliases=["sdxl prompt encode", "sdxl text encode", "sdxl clip encode"],
            inputs=[
                IO.Clip.Input("clip"),
                IO.Int.Input("positive_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("positive_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("negative_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("negative_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("target_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("target_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.String.Input("positive", multiline=True, dynamic_prompts=True),
                IO.String.Input("negative", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                IO.Conditioning.Output("POSITIVE", display_name="POSITIVE"),
                IO.Conditioning.Output("NEGATIVE", display_name="NEGATIVE"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        positive_width: int,
        positive_height: int,
        negative_width: int,
        negative_height: int,
        target_width: int,
        target_height: int,
        positive: str,
        negative: str,
    ) -> IO.NodeOutput:
        tokens_pos = clip.tokenize(positive)
        tokens_neg = clip.tokenize(negative)
        cond_pos = clip.encode_from_tokens_scheduled(
            tokens_pos,
            add_dict={
                "width": positive_width,
                "height": positive_height,
                "target_width": target_width,
                "target_height": target_height,
            },
        )
        cond_neg = clip.encode_from_tokens_scheduled(
            tokens_neg,
            add_dict={
                "width": negative_width,
                "height": negative_height,
                "target_width": target_width,
                "target_height": target_height,
            },
        )
        return IO.NodeOutput(cond_pos, cond_neg)


class PromptTextEncodeSDXLSimple(IO.ComfyNode):
    _POS_DIM_SCALE = 1.5
    _NEG_DIM_SCALE = 0.8

    @staticmethod
    def _get_scaled_dim(target_width: int, target_height: int, scale: float) -> tuple[int, int]:
        scaled_width = round((target_width * scale) / 64) * 64
        scaled_height = round((target_height * scale) / 64) * 64
        return scaled_width, scaled_height

    @staticmethod
    def _get_pos_dim(target_width: int, target_height: int) -> tuple[int, int]:
        return PromptTextEncodeSDXLSimple._get_scaled_dim(target_width, target_height, PromptTextEncodeSDXLSimple._POS_DIM_SCALE)

    @staticmethod
    def _get_neg_dim(target_width: int, target_height: int) -> tuple[int, int]:
        return PromptTextEncodeSDXLSimple._get_scaled_dim(target_width, target_height, PromptTextEncodeSDXLSimple._NEG_DIM_SCALE)

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="SDXL Prompt Text Encode (Simple)",
            category=CATEGORY,
            description="Encodes positive and negative text prompts using a CLIP text encoder with automatically scaled dimensions based on the target resolution.",
            search_aliases=["sdxl prompt encode simple", "sdxl text encode simple", "sdxl clip encode simple"],
            inputs=[
                IO.Clip.Input("clip"),
                IO.Int.Input("target_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.Int.Input("target_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                IO.String.Input("positive", multiline=True, dynamic_prompts=True),
                IO.String.Input("negative", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                IO.Conditioning.Output("POSITIVE", display_name="POSITIVE"),
                IO.Conditioning.Output("NEGATIVE", display_name="NEGATIVE"),
            ],
        )

    @classmethod
    def execute(cls, clip, target_width: int, target_height: int, positive: str, negative: str) -> IO.NodeOutput:
        tokens_pos = clip.tokenize(positive)
        tokens_neg = clip.tokenize(negative)

        width_pos, height_pos = cls._get_pos_dim(target_width, target_height)
        width_neg, height_neg = cls._get_neg_dim(target_width, target_height)

        cond_pos = clip.encode_from_tokens_scheduled(
            tokens_pos,
            add_dict={
                "width": width_pos,
                "height": height_pos,
                "target_width": target_width,
                "target_height": target_height,
            },
        )
        cond_neg = clip.encode_from_tokens_scheduled(
            tokens_neg,
            add_dict={
                "width": width_neg,
                "height": height_neg,
                "target_width": target_width,
                "target_height": target_height,
            },
        )
        return IO.NodeOutput(cond_pos, cond_neg)


CONDITIONING_NODES: list[type[IO.ComfyNode]] = [PromptTextEncode, PromptTextEncodeSDXL, PromptTextEncodeSDXLSimple]
