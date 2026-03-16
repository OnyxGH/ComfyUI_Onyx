from __future__ import annotations

from typing import TypeAlias

NODE_PREFIX = "ONYX"
BASE_CATEGORY = "onyx"

NodeNameSource: TypeAlias = str | type[object]


def _get_class_name(node: NodeNameSource) -> str:
    return node if isinstance(node, str) else node.__name__


def get_node_id(node: NodeNameSource) -> str:
    return f"{NODE_PREFIX}_{_get_class_name(node)}"


def get_category(*parts: str) -> str:
    normalized_parts = [BASE_CATEGORY]
    normalized_parts.extend(
        part.strip(" /") for part in parts if part and part.strip(" /")
    )
    return "/".join(normalized_parts)
