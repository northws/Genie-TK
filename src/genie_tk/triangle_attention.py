'''
Triangle Attention Module

Implements Algorithms 13 & 14 from AlphaFold2:
- Starting Node: Attention along rows with triangle bias
- Ending Node: Attention along columns with triangle bias

These operations compute attention over the pair representation
with bias terms that encode triangular relationships.

License: MIT
'''

from functools import partialmethod
from typing import Optional, List
import math

import torch
import torch.nn as nn
from torch import Tensor

from genie_tk.primitives import Linear, Attention
from genie_tk.utils import has_cuda_kernels, permute_final_dims, chunk_layer

# Try to import CUDA kernels
_CUDA_AVAILABLE = False
try:
    from genie_tk import _C
    _CUDA_AVAILABLE = has_cuda_kernels()
except ImportError:
    pass


class TriangleAttention(nn.Module):
    #
    Triangle Attention (Algorithms 13 & 14 from AlphaFold2).
    
    Computes self-attention over the pair representation with a learned
    triangle bias term that encodes geometric relationships.
    
    Args:
        c_in: Input channel dimension
        c_hidden: Hidden dimension (total, not per-head)
        no_heads: Number of attention heads
        starting: If True, attend along rows (Algorithm 13);
                  if False, attend along columns (Algorithm 14)
        inf: Value for masked positions
        use_fused_kernel: Whether to use ThunderKittens CUDA kernel
    #

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
        use_fused_kernel: bool = True,
    ):
        super().__init__()
        
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.use_fused_kernel = use_fused_kernel and _CUDA_AVAILABLE
        
        # Input layer norm
        self.layer_norm = nn.LayerNorm(c_in)
        
        # Triangle bias projection
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")
        
        # Multi-head attention
        self.mha = Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

    def _compute_triangle_bias(self, x: Tensor) -> Tensor:
        #
        Compute triangle bias from input.
        
        Args:
            x: [*, I, J, C] normalized input
            
        Returns:
            triangle_bias: [*, H, I, J] bias term
        #
        # [*, I, J, H]
        bias = self.linear(x)
        # [*, H, I, J]
        bias = permute_final_dims(bias, 2, 0, 1)
        return bias

    def _forward_pytorch(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        #PyTorch reference implementation.#
        if mask is None:
            mask = x.new_ones(x.shape[:-1], requires_grad=False)
        
        # Handle axis for starting vs ending
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Mask bias: [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        
        # Triangle bias: [*, H, I, J]
        triangle_bias = self._compute_triangle_bias(x)
        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)
        
        # Expand for broadcasting: [*, I, H, I, J]
        seq_len = x.shape[-3]
        triangle_bias = triangle_bias.expand(
            *(-1,) * len(triangle_bias.shape[:-4]), seq_len, -1, -1, -1
        )
        
        # Multi-head attention with biases
        x = self.mha(
            q_x=x,
            k_x=x,
            v_x=x,
            biases=[mask_bias, triangle_bias],
        )
        
        # Transpose back for ending node
        if not self.starting:
            x = x.transpose(-2, -3)
        
        return x

    def _forward_cuda(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        #ThunderKittens CUDA kernel implementation.#
        if mask is None:
            mask = x.new_ones(x.shape[:-1], requires_grad=False)
        
        # Handle axis
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        
        # Layer norm
        x_norm = self.layer_norm(x)
        
        # Compute QKV projections
        # Note: For kernel, we'd pre-compute QKV and pass to kernel
        # For now, we use the MHA module which handles this internally
        
        # Compute triangle bias
        triangle_bias = self._compute_triangle_bias(x_norm)
        
        # Scale factor
        scale = 1.0 / math.sqrt(self.c_hidden // self.no_heads)
        
        # Prepare tensors for kernel
        x_bf16 = x_norm.contiguous().to(torch.bfloat16)
        bias = triangle_bias.contiguous().float()
        mask_2d = mask.contiguous().float()
        
        # For now, fall back to PyTorch (kernel requires QKV pre-projection)
        # TODO: Implement full fused kernel path
        return self._forward_pytorch(x, mask)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        #
        Forward pass.
        
        Args:
            x: Pair representation [*, I, J, c_in]
            mask: Pair mask [*, I, J]
            
        Returns:
            Attended pair representation [*, I, J, c_in]
        #
        if self.use_fused_kernel and x.is_cuda:
            return self._forward_cuda(x, mask)
        else:
            return self._forward_pytorch(x, mask)


class TriangleAttentionStartingNode(TriangleAttention):
    #
    Triangle Attention (Starting Node) - Algorithm 13.
    
    Attention is computed along rows (i-axis). For each row i,
    positions attend to each other along the j-axis.
    #
    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    #
    Triangle Attention (Ending Node) - Algorithm 14.
    
    Attention is computed along columns (j-axis). For each column j,
    positions attend to each other along the i-axis.
    #
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
