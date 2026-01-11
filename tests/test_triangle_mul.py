'''
Tests for Triangle Multiplicative Update

License: MIT
'''

import pytest
import torch
import torch.nn as nn
from typing import Tuple


def make_pair_repr(
    batch: int = 2,
    seq_len: int = 64,
    c_z: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #Create test pair representation and mask.#
    z = torch.randn(batch, seq_len, seq_len, c_z, device=device, dtype=dtype)
    mask = torch.ones(batch, seq_len, seq_len, device=device, dtype=dtype)
    return z, mask


class TestTriangleMultiplicativeUpdate:
    #Test Triangle Multiplicative Update operations.#
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_outgoing_shape(self, device):
        #Test that outgoing update produces correct output shape.#
        from genie_tk import TriangleMultiplicationOutgoing
        
        batch, seq_len, c_z, c_hidden = 2, 32, 64, 32
        z, mask = make_pair_repr(batch, seq_len, c_z, device)
        
        layer = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=False,  # Test PyTorch implementation
        ).to(device)
        
        out = layer(z, mask)
        
        assert out.shape == z.shape
        assert out.device == z.device
        assert out.dtype == z.dtype
    
    def test_incoming_shape(self, device):
        #Test that incoming update produces correct output shape.#
        from genie_tk import TriangleMultiplicationIncoming
        
        batch, seq_len, c_z, c_hidden = 2, 32, 64, 32
        z, mask = make_pair_repr(batch, seq_len, c_z, device)
        
        layer = TriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(z, mask)
        
        assert out.shape == z.shape
    
    def test_gradient_flow(self, device):
        #Test that gradients flow through the layer.#
        from genie_tk import TriangleMultiplicationOutgoing
        
        batch, seq_len, c_z, c_hidden = 1, 16, 32, 16
        z, mask = make_pair_repr(batch, seq_len, c_z, device)
        z.requires_grad_(True)
        
        layer = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(z, mask)
        loss = out.sum()
        loss.backward()
        
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
    
    def test_mask_application(self, device):
        #Test that mask is properly applied.#
        from genie_tk import TriangleMultiplicationOutgoing
        
        batch, seq_len, c_z, c_hidden = 1, 16, 32, 16
        z, _ = make_pair_repr(batch, seq_len, c_z, device)
        
        # Create mask with some zeros
        mask = torch.ones(batch, seq_len, seq_len, device=device)
        mask[:, :, seq_len//2:] = 0  # Mask out half
        
        layer = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=False,
        ).to(device)
        
        out = layer(z, mask)
        
        # Output should be computed but we mainly test it doesn't error
        assert not torch.isnan(out).any()


class TestTriangleMultiplicativeUpdateCUDA:
    #Test CUDA kernel implementation (if available).#
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return "cuda"
    
    def test_cuda_vs_pytorch_outgoing(self, device):
        #Compare CUDA and PyTorch implementations for outgoing.#
        from genie_tk import TriangleMultiplicationOutgoing
        from genie_tk.utils import has_cuda_kernels
        
        if not has_cuda_kernels():
            pytest.skip("CUDA kernels not compiled")
        
        batch, seq_len, c_z, c_hidden = 1, 32, 64, 32
        z, mask = make_pair_repr(batch, seq_len, c_z, device)
        
        # PyTorch implementation
        layer_pt = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=False,
        ).to(device)
        
        # CUDA implementation (same weights)
        layer_cuda = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden,
            use_fused_kernel=True,
        ).to(device)
        layer_cuda.load_state_dict(layer_pt.state_dict())
        
        out_pt = layer_pt(z, mask)
        out_cuda = layer_cuda(z, mask)
        
        # Should be close (may have small numerical differences)
        torch.testing.assert_close(out_pt, out_cuda, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
