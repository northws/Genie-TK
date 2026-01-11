'''
Pair Transform Module

Combines Triangle operations into a complete pair transform layer/network.

License: MIT
'''

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from genie_tk.triangle_mul import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from genie_tk.triangle_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from genie_tk.primitives import Linear


class DropoutRowwise(nn.Module):
    #Apply dropout with same mask across rows.#
    
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        # x: [*, L, L, C] - same dropout mask for each row
        shape = list(x.shape)
        shape[-3] = 1  # Broadcast over rows
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        return x * mask


class DropoutColumnwise(nn.Module):
    #Apply dropout with same mask across columns.#
    
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        # x: [*, L, L, C] - same dropout mask for each column
        shape = list(x.shape)
        shape[-2] = 1  # Broadcast over columns
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        return x * mask


class PairTransition(nn.Module):
    #
    Pair Transition layer (feed-forward network for pair representation).
    #
    
    def __init__(self, c_p: int, n: int = 4):
        super().__init__()
        self.c_p = c_p
        self.n = n
        
        self.layer_norm = nn.LayerNorm(c_p)
        self.linear_1 = Linear(c_p, c_p * n)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(c_p * n, c_p, init="final")
    
    def forward(self, p: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        p = self.layer_norm(p)
        p = self.linear_1(p)
        p = self.relu(p)
        p = self.linear_2(p)
        return p


class PairTransformLayer(nn.Module):
    #
    Single Pair Transform Layer.
    
    Combines Triangle operations for updating pair representations:
    1. Triangle Multiplicative Update (Outgoing)
    2. Triangle Multiplicative Update (Incoming)
    3. Triangle Attention (Starting Node)
    4. Triangle Attention (Ending Node)
    5. Pair Transition (Feed-forward)
    
    Args:
        c_p: Pair representation dimension
        include_mul_update: Whether to include triangle multiplicative updates
        include_tri_att: Whether to include triangle attention
        c_hidden_mul: Hidden dimension for multiplicative updates
        c_hidden_tri_att: Hidden dimension for triangle attention
        n_head_tri: Number of attention heads
        tri_dropout: Dropout probability
        pair_transition_n: Expansion factor for pair transition
        use_fused_kernel: Whether to use ThunderKittens kernels
    #

    def __init__(
        self,
        c_p: int,
        include_mul_update: bool = True,
        include_tri_att: bool = True,
        c_hidden_mul: int = 128,
        c_hidden_tri_att: int = 32,
        n_head_tri: int = 4,
        tri_dropout: float = 0.25,
        pair_transition_n: int = 4,
        use_fused_kernel: bool = True,
    ):
        super().__init__()
        
        self.include_mul_update = include_mul_update
        self.include_tri_att = include_tri_att
        
        # Triangle Multiplicative Updates
        if include_mul_update:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z=c_p,
                c_hidden=c_hidden_mul,
                use_fused_kernel=use_fused_kernel,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z=c_p,
                c_hidden=c_hidden_mul,
                use_fused_kernel=use_fused_kernel,
            )
        
        # Triangle Attention
        if include_tri_att:
            self.tri_att_start = TriangleAttentionStartingNode(
                c_in=c_p,
                c_hidden=c_hidden_tri_att,
                no_heads=n_head_tri,
                use_fused_kernel=use_fused_kernel,
            )
            self.tri_att_end = TriangleAttentionEndingNode(
                c_in=c_p,
                c_hidden=c_hidden_tri_att,
                no_heads=n_head_tri,
                use_fused_kernel=use_fused_kernel,
            )
        
        # Pair Transition
        self.pair_transition = PairTransition(c_p, pair_transition_n)
        
        # Dropout layers
        self.dropout_row = DropoutRowwise(tri_dropout)
        self.dropout_col = DropoutColumnwise(tri_dropout)

    def forward(
        self,
        inputs: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        #
        Forward pass.
        
        Args:
            inputs: Tuple of (p, p_mask) where
                p: Pair representation [B, L, L, c_p]
                p_mask: Pair mask [B, L, L]
                
        Returns:
            Tuple of (p, p_mask) - updated pair representation
        #
        p, p_mask = inputs
        
        # Triangle Multiplicative Updates with residual connections
        if self.include_mul_update:
            p = p + self.dropout_row(self.tri_mul_out(p, p_mask))
            p = p + self.dropout_row(self.tri_mul_in(p, p_mask))
        
        # Triangle Attention with residual connections
        if self.include_tri_att:
            p = p + self.dropout_row(self.tri_att_start(p, p_mask))
            p = p + self.dropout_col(self.tri_att_end(p, p_mask))
        
        # Pair Transition with residual connection
        p = p + self.pair_transition(p, p_mask)
        
        # Apply mask
        p = p * p_mask.unsqueeze(-1)
        
        return (p, p_mask)


class PairTransformNet(nn.Module):
    #
    Pair Transform Network.
    
    Stack of PairTransformLayers for iterative refinement of pair representations.
    
    Args:
        c_p: Pair representation dimension
        n_layers: Number of transform layers
        **kwargs: Arguments passed to PairTransformLayer
    #

    def __init__(
        self,
        c_p: int,
        n_layers: int,
        include_mul_update: bool = True,
        include_tri_att: bool = True,
        c_hidden_mul: int = 128,
        c_hidden_tri_att: int = 32,
        n_head_tri: int = 4,
        tri_dropout: float = 0.25,
        pair_transition_n: int = 4,
        use_fused_kernel: bool = True,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            PairTransformLayer(
                c_p=c_p,
                include_mul_update=include_mul_update,
                include_tri_att=include_tri_att,
                c_hidden_mul=c_hidden_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                n_head_tri=n_head_tri,
                tri_dropout=tri_dropout,
                pair_transition_n=pair_transition_n,
                use_fused_kernel=use_fused_kernel,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        p: Tensor,
        p_mask: Tensor,
    ) -> Tensor:
        #
        Forward pass.
        
        Args:
            p: Pair representation [B, L, L, c_p]
            p_mask: Pair mask [B, L, L]
            
        Returns:
            Updated pair representation [B, L, L, c_p]
        #
        for layer in self.layers:
            p, p_mask = layer((p, p_mask))
        
        return p
