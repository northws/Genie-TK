'''
Tests for Triangle Attention

License: MIT
'''

import pytest
import torch
from typing import Tuple


def make_pair_repr(
    batch: int = 2,
    seq_len: int = 64,
    c_in: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #Create test pair representation and mask.#
    x = torch.randn(batch, seq_len, seq_len, c_in, device=device, dtype=dtype)
    mask = torch.ones(batch, seq_len, seq_len, device=device, dtype=dtype)
    return x, mask


class TestTriangleAttention:
    #Test Triangle Attention operations.#
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_starting_shape(self, device):
        #Test that starting node attention produces correct output shape.#
        from genie_tk import TriangleAttentionStartingNode
        
        batch, seq_len, c_in, c_hidden, no_heads = 2, 32, 64, 32, 4
        x, mask = make_pair_repr(batch, seq_len, c_in, device)
        
        layer = TriangleAttentionStartingNode(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(x, mask)
        
        assert out.shape == x.shape
        assert out.device == x.device
        assert out.dtype == x.dtype
    
    def test_ending_shape(self, device):
        #Test that ending node attention produces correct output shape.#
        from genie_tk import TriangleAttentionEndingNode
        
        batch, seq_len, c_in, c_hidden, no_heads = 2, 32, 64, 32, 4
        x, mask = make_pair_repr(batch, seq_len, c_in, device)
        
        layer = TriangleAttentionEndingNode(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(x, mask)
        
        assert out.shape == x.shape
    
    def test_gradient_flow(self, device):
        #Test that gradients flow through the layer.#
        from genie_tk import TriangleAttentionStartingNode
        
        batch, seq_len, c_in, c_hidden, no_heads = 1, 16, 32, 16, 2
        x, mask = make_pair_repr(batch, seq_len, c_in, device)
        x.requires_grad_(True)
        
        layer = TriangleAttentionStartingNode(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(x, mask)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_attention_mask(self, device):
        #Test that attention mask is properly applied.#
        from genie_tk import TriangleAttentionStartingNode
        
        batch, seq_len, c_in, c_hidden, no_heads = 1, 16, 32, 16, 2
        x, _ = make_pair_repr(batch, seq_len, c_in, device)
        
        # Create mask with some zeros
        mask = torch.ones(batch, seq_len, seq_len, device=device)
        mask[:, :, seq_len//2:] = 0
        
        layer = TriangleAttentionStartingNode(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(x, mask)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestPairTransform:
    #Test complete Pair Transform layer.#
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_pair_transform_layer(self, device):
        #Test PairTransformLayer.#
        from genie_tk import PairTransformLayer
        
        batch, seq_len, c_p = 1, 32, 64
        p = torch.randn(batch, seq_len, seq_len, c_p, device=device)
        p_mask = torch.ones(batch, seq_len, seq_len, device=device)
        
        layer = PairTransformLayer(
            c_p=c_p,
            include_mul_update=True,
            include_tri_att=True,
            c_hidden_mul=32,
            c_hidden_tri_att=16,
            n_head_tri=4,
            tri_dropout=0.0,  # Disable dropout for testing
            use_fused_kernel=False,
        ).to(device)
        
        out_p, out_mask = layer((p, p_mask))
        
        assert out_p.shape == p.shape
        assert not torch.isnan(out_p).any()
    
    def test_pair_transform_net(self, device):
        #Test PairTransformNet.#
        from genie_tk import PairTransformNet
        
        batch, seq_len, c_p = 1, 32, 64
        p = torch.randn(batch, seq_len, seq_len, c_p, device=device)
        p_mask = torch.ones(batch, seq_len, seq_len, device=device)
        
        net = PairTransformNet(
            c_p=c_p,
            n_layers=2,
            include_mul_update=True,
            include_tri_att=True,
            c_hidden_mul=32,
            c_hidden_tri_att=16,
            n_head_tri=4,
            tri_dropout=0.0,
            use_fused_kernel=False,
        ).to(device)
        
        out_p = net(p, p_mask)
        
        assert out_p.shape == p.shape
        assert not torch.isnan(out_p).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
