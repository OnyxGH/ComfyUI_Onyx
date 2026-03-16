from .bundle_nodes import BUNDLE_NODES
from .loader_nodes import LOADER_NODES
from .math_nodes import MATH_NODES
from .logic_nodes import LOGIC_NODES
from .model.model_merge_nodes import MODEL_MERGE_NODES
from .model.model_patch_nodes import MODEL_PATCH_NODES
from .primitive_nodes import PRIMITIVE_NODES
from .widget_nodes import WIDGET_NODES

NODE_LIST = [
    *BUNDLE_NODES,
    *MATH_NODES,
    *WIDGET_NODES,
    *PRIMITIVE_NODES,
    *LOGIC_NODES,
    *LOADER_NODES,
    *MODEL_PATCH_NODES,
    *MODEL_MERGE_NODES,
]
