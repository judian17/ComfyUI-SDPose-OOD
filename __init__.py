# Import the mappings from your nodes file (sdpose_ood_nodes.py)
from .sdpose_ood_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose the mappings for ComfyUI to discover
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("### Loading: SDPose OOD Nodes ###")
