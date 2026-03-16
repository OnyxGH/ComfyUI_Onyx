from comfy_api.latest import IO

from ..lib.bundle import CATEGORY, build_bundle_nodes

BUNDLE_NODES: list[type[IO.ComfyNode]] = build_bundle_nodes(category=CATEGORY)
NODE_LIST = BUNDLE_NODES
