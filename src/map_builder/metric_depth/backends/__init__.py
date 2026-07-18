from .depth_anything_v2 import DepthAnythingV2AlignedBackend
from .prompt_depth_anything import PromptDepthAnythingBackend

BACKENDS = {
    "depth_anything_v2_aligned": DepthAnythingV2AlignedBackend,
    "prompt_depth_anything": PromptDepthAnythingBackend,
}

__all__ = ["BACKENDS", "DepthAnythingV2AlignedBackend", "PromptDepthAnythingBackend"]
