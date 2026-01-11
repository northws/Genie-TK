'''
Genie-TK: ThunderKittens-Accelerated Triangle Operations

High-performance CUDA kernels for protein structure prediction,
implementing Triangle operations from AlphaFold2/Genie.

License: MIT
'''

__version__ = "0.1.0"
__author__ = "Genie-TK Contributors"

from genie_tk.triangle_mul import (
    TriangleMultiplicativeUpdate,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)

from genie_tk.triangle_attention import (
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)

from genie_tk.pair_transform import (
    PairTransformLayer,
    PairTransformNet,
)

from genie_tk.utils import (
    has_cuda_kernels,
    get_device_info,
)

__all__ = [
    # Triangle Multiplicative Update
    "TriangleMultiplicativeUpdate",
    "TriangleMultiplicationOutgoing",
    "TriangleMultiplicationIncoming",
    # Triangle Attention
    "TriangleAttention",
    "TriangleAttentionStartingNode",
    "TriangleAttentionEndingNode",
    # Pair Transform
    "PairTransformLayer",
    "PairTransformNet",
    # Utils
    "has_cuda_kernels",
    "get_device_info",
]
