from comfy_api.latest import IO

from ..helpers.nodes import get_node_id
from ..lib.bundle import CATEGORY, BundleType, build_bundle_nodes, merge_bundles


class BundleMerge(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=get_node_id(cls),
            display_name="Bundle Merge",
            category=CATEGORY,
            description="Merges multiple bundles into one. Later bundles will overwrite keys from earlier bundles if there are conflicts.",
            search_aliases=["bundle merge", "merge bundles", "combine bundles"],
            inputs=[
                IO.Autogrow.Input("bundles", template=IO.Autogrow.TemplatePrefix(BundleType.Input("bundle"), prefix="bundle_", min=2, max=50)),
            ],
            outputs=[
                BundleType.Output("BUNDLE", display_name="BUNDLE"),
            ],
        )

    @classmethod
    def execute(cls, bundles: IO.Autogrow.Type) -> IO.NodeOutput:
        merged_bundle = merge_bundles(bundle for bundle in bundles.values() if bundle is not None)
        return IO.NodeOutput(merged_bundle)


BUNDLE_NODES: list[type[IO.ComfyNode]] = [*build_bundle_nodes(category=CATEGORY), BundleMerge]
