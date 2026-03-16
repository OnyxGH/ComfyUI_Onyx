from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .modules import NODE_LIST


class OnyxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return NODE_LIST


async def comfy_entrypoint() -> OnyxExtension:
    return OnyxExtension()


__all__ = ["OnyxExtension", "comfy_entrypoint"]
