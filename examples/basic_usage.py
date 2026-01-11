'''
Example: Using Genie-TK for protein structure prediction

This example demonstrates how to use the Triangle operations
from Genie-TK in a protein structure prediction pipeline.

License: MIT
'''

import torch
import torch.nn as nn
from typing import Tuple, Optional


def example_basic_usage():
    #Basic usage of Triangle operations.#
    from genie_tk import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming,
        TriangleAttentionStartingNode,
        TriangleAttentionEndingNode,
    )
    
    # Configuration
    batch_size = 1
    seq_len = 64
    c_z = 128  # Pair representation dimension
    c_hidden = 128  # Hidden dimension for projections
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running on {device}")
    
    # Create pair representation
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device)
    mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    
    # Triangle Multiplicative Update (Outgoing)
    print("Creating Triangle Multiplicative Update (Outgoing)...")
    tri_mul_out = TriangleMultiplicationOutgoing(
        c_z=c_z,
        c_hidden=c_hidden,
    ).to(device)
    
    z_out = tri_mul_out(z, mask)
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {z_out.shape}")
    
    # Apply with residual connection
    z = z + z_out
    
    # Triangle Multiplicative Update (Incoming)
    print("Creating Triangle Multiplicative Update (Incoming)...")
    tri_mul_in = TriangleMultiplicationIncoming(
        c_z=c_z,
        c_hidden=c_hidden,
    ).to(device)
    
    z = z + tri_mul_in(z, mask)
    
    # Triangle Attention (Starting Node)
    print("Creating Triangle Attention (Starting Node)...")
    tri_att_start = TriangleAttentionStartingNode(
        c_in=c_z,
        c_hidden=32,
        no_heads=4,
    ).to(device)
    
    z = z + tri_att_start(z, mask)
    
    # Triangle Attention (Ending Node)
    print("Creating Triangle Attention (Ending Node)...")
    tri_att_end = TriangleAttentionEndingNode(
        c_in=c_z,
        c_hidden=32,
        no_heads=4,
    ).to(device)
    
    z = z + tri_att_end(z, mask)
    
    print(f"\nFinal pair representation shape: {z.shape}")
    print("Done!")


def example_pair_transform_net():
    #Using the complete PairTransformNet.#
    from genie_tk import PairTransformNet
    
    batch_size = 1
    seq_len = 64
    c_p = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Creating PairTransformNet on {device}...")
    
    # Create pair representation
    p = torch.randn(batch_size, seq_len, seq_len, c_p, device=device)
    p_mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    
    # Create network
    net = PairTransformNet(
        c_p=c_p,
        n_layers=4,
        include_mul_update=True,
        include_tri_att=True,
        c_hidden_mul=128,
        c_hidden_tri_att=32,
        n_head_tri=4,
        tri_dropout=0.0,  # Disable for inference
        use_fused_kernel=True,  # Use CUDA kernels if available
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        p_out = net(p, p_mask)
    
    print(f"Input shape: {p.shape}")
    print(f"Output shape: {p_out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters()):,}")


def example_gradient_checkpointing():
    #Using gradient checkpointing for memory efficiency.#
    from torch.utils.checkpoint import checkpoint
    from genie_tk import PairTransformLayer
    
    batch_size = 1
    seq_len = 128  # Longer sequence
    c_p = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing gradient checkpointing on {device}...")
    
    p = torch.randn(batch_size, seq_len, seq_len, c_p, device=device, requires_grad=True)
    p_mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    
    layer = PairTransformLayer(
        c_p=c_p,
        include_mul_update=True,
        include_tri_att=True,
        c_hidden_mul=64,
        c_hidden_tri_att=16,
        n_head_tri=4,
        tri_dropout=0.0,
    ).to(device)
    
    # With gradient checkpointing
    def forward_fn(p, p_mask):
        return layer((p, p_mask))[0]
    
    p_out = checkpoint(forward_fn, p, p_mask, use_reentrant=False)
    
    # Backward pass
    loss = p_out.sum()
    loss.backward()
    
    print(f"Gradient computed successfully!")
    print(f"Gradient norm: {p.grad.norm().item():.4f}")


def example_integration_with_genie():
    #Example of integrating with a Genie-like model.#
    from genie_tk import PairTransformNet
    
    class SimplePairModule(nn.Module):
        #Simplified pair representation module.#
        
        def __init__(self, c_s: int, c_p: int, n_layers: int):
            super().__init__()
            
            # Initial pair embedding
            self.pair_embed = nn.Linear(c_s * 2, c_p)
            
            # Pair transform network
            self.pair_transform = PairTransformNet(
                c_p=c_p,
                n_layers=n_layers,
                include_mul_update=True,
                include_tri_att=True,
            )
        
        def forward(
            self,
            s: torch.Tensor,  # [B, L, c_s] single representation
            mask: torch.Tensor,  # [B, L]
        ) -> torch.Tensor:
            B, L, _ = s.shape
            
            # Create initial pair embedding from outer product
            # p_ij = [s_i; s_j]
            s_i = s.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, c_s]
            s_j = s.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, c_s]
            p = torch.cat([s_i, s_j], dim=-1)  # [B, L, L, 2*c_s]
            p = self.pair_embed(p)  # [B, L, L, c_p]
            
            # Create pair mask
            p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, L, L]
            
            # Transform pair representation
            p = self.pair_transform(p, p_mask)
            
            return p
    
    # Usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SimplePairModule(c_s=64, c_p=128, n_layers=2).to(device)
    
    # Input
    s = torch.randn(1, 32, 64, device=device)  # Single representation
    mask = torch.ones(1, 32, device=device)  # Sequence mask
    
    # Forward
    with torch.no_grad():
        p = model(s, mask)
    
    print(f"Single repr shape: {s.shape}")
    print(f"Pair repr shape: {p.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    example_basic_usage()
    
    print("\n" + "=" * 60)
    print("Example 2: PairTransformNet")
    print("=" * 60)
    example_pair_transform_net()
    
    print("\n" + "=" * 60)
    print("Example 3: Gradient Checkpointing")
    print("=" * 60)
    example_gradient_checkpointing()
    
    print("\n" + "=" * 60)
    print("Example 4: Integration Example")
    print("=" * 60)
    example_integration_with_genie()
