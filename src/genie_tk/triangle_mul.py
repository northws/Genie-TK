'''
Triangle Multiplicative Update Module

Implements Algorithms 11 & 12 from AlphaFold2:
- Outgoing: Z_ij += Σ_k (a_ik ⊗ b_jk)
- Incoming: Z_ij += Σ_k (a_ki ⊗ b_kj)

These operations enforce geometric consistency in the pair representation
by propagating information along triangle edges.

License: MIT
'''

from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from genie_tk.primitives import Linear
from genie_tk.utils import has_cuda_kernels, permute_final_dims

# Try to import CUDA kernels
_CUDA_AVAILABLE = False
try:
    from genie_tk import _C
    _CUDA_AVAILABLE = has_cuda_kernels()
except ImportError:
    pass


class TriangleMultiplicativeUpdate(nn.Module):
    #
    Triangle Multiplicative Update (Algorithms 11 & 12 from AlphaFold2).
    
    This module updates the pair representation by aggregating information
    along triangle edges. For each pair (i,j), it considers all residues k
    that form triangles with i and j.
    
    Args:
        c_z: Input/output channel dimension
        c_hidden: Hidden channel dimension for projections
        _outgoing: If True, use outgoing update (Algorithm 11);
                   if False, use incoming update (Algorithm 12)
        use_fused_kernel: Whether to use ThunderKittens CUDA kernel
    #

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        _outgoing: bool = True,
        use_fused_kernel: bool = True,
    ):
        super().__init__()
        
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.use_fused_kernel = use_fused_kernel and _CUDA_AVAILABLE
        
        # Linear projections
        self.linear_a_p = Linear(c_z, c_hidden)
        self.linear_a_g = Linear(c_z, c_hidden, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden)
        self.linear_b_g = Linear(c_z, c_hidden, init="gating")
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.linear_z = Linear(c_hidden, c_z, init="final")
        
        # Layer norms
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        
        self.sigmoid = nn.Sigmoid()

    def _outgoing_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        #
        Outgoing triangle: Z_ij = Σ_k a_ik * b_jk
        Equivalent to: einsum('bikc,bjkc->bijc', a, b)
        #
        # a: [B, I, K, C], b: [B, J, K, C]
        # We want: [B, I, J, C] where output[i,j,c] = sum_k a[i,k,c] * b[j,k,c]
        # This is a @ b^T per channel
        p = torch.matmul(
            permute_final_dims(a, 2, 0, 1),  # [B, C, I, K]
            permute_final_dims(b, 2, 1, 0),  # [B, C, K, J]
        )  # [B, C, I, J]
        return permute_final_dims(p, 1, 2, 0)  # [B, I, J, C]

    def _incoming_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        #
        Incoming triangle: Z_ij = Σ_k a_ki * b_kj
        Equivalent to: einsum('bkic,bkjc->bijc', a, b)
        #
        # a: [B, K, I, C], b: [B, K, J, C]
        p = torch.matmul(
            permute_final_dims(a, 2, 1, 0),  # [B, C, K, I] -> transpose to [B, C, I, K]
            permute_final_dims(b, 2, 0, 1),  # [B, C, K, J]
        )
        return permute_final_dims(p, 1, 2, 0)

    def _forward_pytorch(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        #PyTorch reference implementation.#
        if mask is None:
            mask = z.new_ones(z.shape[:-1], requires_grad=False)
        
        mask = mask.unsqueeze(-1)
        
        # Input LayerNorm
        z = self.layer_norm_in(z)
        
        # Compute gated projections
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask
        
        # Triangle multiplication
        if self._outgoing:
            x = self._outgoing_matmul(a, b)
        else:
            x = self._incoming_matmul(a, b)
        
        # Output processing
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        
        # Output gating
        g = self.sigmoid(self.linear_g(z))
        
        return x * g

    def _forward_cuda(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        #ThunderKittens CUDA kernel implementation.#
        if mask is None:
            mask = z.new_ones(z.shape[:-1], requires_grad=False)
        
        mask = mask.unsqueeze(-1)
        
        # Input LayerNorm
        z_norm = self.layer_norm_in(z)
        
        # Compute gated projections
        a = self.linear_a_p(z_norm) * self.sigmoid(self.linear_a_g(z_norm))
        a = a * mask
        b = self.linear_b_p(z_norm) * self.sigmoid(self.linear_b_g(z_norm))
        b = b * mask
        
        # Ensure contiguous bf16 for kernel
        a = a.contiguous().to(torch.bfloat16)
        b = b.contiguous().to(torch.bfloat16)
        mask_2d = mask.squeeze(-1).contiguous().float()
        
        # Call CUDA kernel
        if self._outgoing:
            x = _C.triangle_mul_outgoing(a, b, mask_2d)
        else:
            x = _C.triangle_mul_incoming(a, b, mask_2d)
        
        # Convert back and apply output processing
        x = x.to(z.dtype)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        
        # Output gating
        g = self.sigmoid(self.linear_g(z_norm))
        
        return x * g

    def forward(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        #
        Forward pass.
        
        Args:
            z: Pair representation [B, L, L, c_z]
            mask: Pair mask [B, L, L]
            
        Returns:
            Updated pair representation [B, L, L, c_z]
        #
        if self.use_fused_kernel and z.is_cuda:
            return self._forward_cuda(z, mask)
        else:
            return self._forward_pytorch(z, mask)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    #
    Triangle Multiplicative Update (Outgoing) - Algorithm 11.
    
    Information flows from residues i and j to their common neighbor k:
    Z_ij += Σ_k (a_ik ⊗ b_jk)
    #
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    #
    Triangle Multiplicative Update (Incoming) - Algorithm 12.
    
    Information flows from common neighbor k to residues i and j:
    Z_ij += Σ_k (a_ki ⊗ b_kj)
    #
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
