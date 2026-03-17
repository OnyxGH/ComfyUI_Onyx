from __future__ import annotations

from comfy_extras.nodes_model_merging import CLIPMergeSimple, ModelMergeSimple


def merge_checkpoints(model_a, model_b, clip_a, clip_b, model_ratio, clip_ratio):
    merged_model = ModelMergeSimple().merge(model_a, model_b, model_ratio)[0]
    merged_clip = CLIPMergeSimple().merge(clip_a, clip_b, clip_ratio)[0]
    return merged_model, merged_clip
