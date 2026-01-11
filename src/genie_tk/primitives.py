'''
Primitive Modules for Genie-TK

Basic building blocks used across the library.

License: MIT
'''

from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn
from torch import Tensor


def _init_weights(
    linear: nn.Linear,
    init: str = "default",
    std: float = 0.02,
):
    #Initialize linear layer weights.#
    if init == "default":
        nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        if linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)
    elif init == "final":
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    elif init == "gating":
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.ones_(linear.bias)
    elif init == "normal":
        nn.init.normal_(linear.weight, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


class Linear(nn.Linear):
    #
    Linear layer with specialized initialization.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        init: Initialization method ('default', 'final', 'gating', 'normal')
    #

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
    ):
        super().__init__(in_features, out_features, bias=bias)
        _init_weights(self, init)


class Attention(nn.Module):
    #
    Multi-head attention with bias support.
    
    Args:
        c_q: Query input dimension
        c_k: Key input dimension
        c_v: Value input dimension
        c_hidden: Hidden dimension (total across all heads)
        no_heads: Number of attention heads
        gating: Whether to apply output gating
    #

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        super().__init__()
        
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        
        assert c_hidden % no_heads == 0
        self.head_dim = c_hidden // no_heads
        
        # QKV projections
        self.linear_q = Linear(c_q, c_hidden, bias=False)
        self.linear_k = Linear(c_k, c_hidden, bias=False)
        self.linear_v = Linear(c_v, c_hidden, bias=False)
        
        # Output projection
        self.linear_o = Linear(c_hidden, c_q, init="final")
        
        # Gating
        if gating:
            self.linear_g = Linear(c_q, c_hidden, init="gating")
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q_x: Tensor,
        k_x: Tensor,
        v_x: Tensor,
        biases: Optional[List[Tensor]] = None,
    ) -> Tensor:
        #
        Forward pass.
        
        Args:
            q_x: Query input [*, Q, c_q]
            k_x: Key input [*, K, c_k]
            v_x: Value input [*, K, c_v]
            biases: List of bias tensors to add to attention scores
            
        Returns:
            Output tensor [*, Q, c_q]
        #
        # Project to QKV
        q = self.linear_q(q_x)  # [*, Q, H*D]
        k = self.linear_k(k_x)  # [*, K, H*D]
        v = self.linear_v(v_x)  # [*, K, H*D]
        
        # Reshape for multi-head attention
        *batch_dims, q_len, _ = q.shape
        *_, k_len, _ = k.shape
        
        q = q.view(*batch_dims, q_len, self.no_heads, self.head_dim)
        k = k.view(*batch_dims, k_len, self.no_heads, self.head_dim)
        v = v.view(*batch_dims, k_len, self.no_heads, self.head_dim)
        
        # Transpose to [*, H, Q/K, D]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # [*, H, Q, K]
        
        # Add biases
        if biases is not None:
            for bias in biases:
                attn = attn + bias
        
        # Softmax
        attn = self.softmax(attn)
        
        # Apply attention to values
        o = torch.matmul(attn, v)  # [*, H, Q, D]
        
        # Transpose back
        o = o.transpose(-2, -3)  # [*, Q, H, D]
        o = o.reshape(*batch_dims, q_len, self.c_hidden)  # [*, Q, H*D]
        
        # Gating
        if self.gating:
            g = self.sigmoid(self.linear_g(q_x))
            o = o * g
        
        # Output projection
        o = self.linear_o(o)
        
        return o
