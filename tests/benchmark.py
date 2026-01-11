'''
Benchmarks for Genie-TK operations

Compares performance of ThunderKittens kernels vs PyTorch implementations.

License: MIT
'''

import time
from typing import Callable, Dict, Any
import torch
import argparse


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs,
) -> Dict[str, float]:
    #Benchmark a function.
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "avg_time": elapsed / iterations,
        "iterations": iterations,
    }


def benchmark_triangle_mul(
    batch: int,
    seq_len: int,
    c_z: int,
    c_hidden: int,
    device: str,
    use_cuda_kernel: bool,
):
    #Benchmark Triangle Multiplicative Update.
    from genie_tk import TriangleMultiplicationOutgoing
    
    z = torch.randn(batch, seq_len, seq_len, c_z, device=device, dtype=torch.float32)
    mask = torch.ones(batch, seq_len, seq_len, device=device)
    
    layer = TriangleMultiplicationOutgoing(
        c_z=c_z,
        c_hidden=c_hidden,
        use_fused_kernel=use_cuda_kernel,
    ).to(device)
    
    # Forward benchmark
    def forward_fn():
        return layer(z, mask)
    
    result = benchmark_fn(forward_fn)
    
    return result


def benchmark_triangle_attention(
    batch: int,
    seq_len: int,
    c_in: int,
    c_hidden: int,
    no_heads: int,
    device: str,
    use_cuda_kernel: bool,
):
    #Benchmark Triangle Attention.
    from genie_tk import TriangleAttentionStartingNode
    
    x = torch.randn(batch, seq_len, seq_len, c_in, device=device, dtype=torch.float32)
    mask = torch.ones(batch, seq_len, seq_len, device=device)
    
    layer = TriangleAttentionStartingNode(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads,
        use_fused_kernel=use_cuda_kernel,
    ).to(device)
    
    def forward_fn():
        return layer(x, mask)
    
    result = benchmark_fn(forward_fn)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Genie-TK operations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--c-z", type=int, default=128, help="Pair representation dimension")
    parser.add_argument("--c-hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    print(f"Benchmarking on {device}")
    print(f"Batch: {args.batch}, SeqLen: {args.seq_len}, C_z: {args.c_z}")
    print("=" * 60)
    
    # Triangle Multiplicative Update - PyTorch
    print("\nTriangle Multiplicative Update (Outgoing)")
    result_pt = benchmark_triangle_mul(
        args.batch, args.seq_len, args.c_z, args.c_hidden, device, False
    )
    print(f"  PyTorch:  {result_pt['avg_time']*1000:.3f} ms")
    
    # Triangle Multiplicative Update - CUDA (if available)
    from genie_tk.utils import has_cuda_kernels
    if has_cuda_kernels() and device == "cuda":
        result_cuda = benchmark_triangle_mul(
            args.batch, args.seq_len, args.c_z, args.c_hidden, device, True
        )
        print(f"  CUDA:     {result_cuda['avg_time']*1000:.3f} ms")
        print(f"  Speedup:  {result_pt['avg_time']/result_cuda['avg_time']:.2f}x")
    
    # Triangle Attention
    print("\nTriangle Attention (Starting Node)")
    result_pt = benchmark_triangle_attention(
        args.batch, args.seq_len, args.c_z, 32, 4, device, False
    )
    print(f"  PyTorch:  {result_pt['avg_time']*1000:.3f} ms")
    
    if has_cuda_kernels() and device == "cuda":
        result_cuda = benchmark_triangle_attention(
            args.batch, args.seq_len, args.c_z, 32, 4, device, True
        )
        print(f"  CUDA:     {result_cuda['avg_time']*1000:.3f} ms")
        print(f"  Speedup:  {result_pt['avg_time']/result_cuda['avg_time']:.2f}x")


if __name__ == "__main__":
    main()
