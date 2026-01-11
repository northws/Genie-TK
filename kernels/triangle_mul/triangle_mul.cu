/**
 * Genie-TK: Triangle Multiplicative Update CUDA Implementation
 * 
 * This file contains the CUDA kernel implementations and PyTorch bindings
 * for the Triangle Multiplicative Update operations.
 * 
 * License: MIT
 */

#include "triangle_mul.cuh"
#include "pyutils/pyutils.cuh"

namespace genie_tk {
namespace triangle_mul {

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

template<int C_HIDDEN>
void launch_triangle_outgoing(TriangleOutgoingGlobals<C_HIDDEN>& g, cudaStream_t stream = 0) {
    // Calculate shared memory requirement
    size_t smem_size = sizeof(smem_tile_bf16) * 2 * STAGES + sizeof(smem_acc_tile);
    
    // Launch kernel
    triangle_outgoing_kernel<C_HIDDEN><<<g.grid(), g.block(), smem_size, stream>>>(g);
}

template<int C_HIDDEN>
void launch_triangle_incoming(TriangleIncomingGlobals<C_HIDDEN>& g, cudaStream_t stream = 0) {
    size_t smem_size = sizeof(smem_tile_bf16) * 2 * STAGES + sizeof(smem_acc_tile);
    triangle_incoming_kernel<C_HIDDEN><<<g.grid(), g.block(), smem_size, stream>>>(g);
}

// Explicit instantiations for common hidden dimensions
template void launch_triangle_outgoing<64>(TriangleOutgoingGlobals<64>&, cudaStream_t);
template void launch_triangle_outgoing<128>(TriangleOutgoingGlobals<128>&, cudaStream_t);
template void launch_triangle_outgoing<256>(TriangleOutgoingGlobals<256>&, cudaStream_t);

template void launch_triangle_incoming<64>(TriangleIncomingGlobals<64>&, cudaStream_t);
template void launch_triangle_incoming<128>(TriangleIncomingGlobals<128>&, cudaStream_t);
template void launch_triangle_incoming<256>(TriangleIncomingGlobals<256>&, cudaStream_t);

} // namespace triangle_mul
} // namespace genie_tk

// ============================================================================
// Python Bindings via ThunderKittens PyUtils
// ============================================================================

using namespace genie_tk::triangle_mul;

// Wrapper functions for Python binding
void run_triangle_outgoing_128(TriangleOutgoingGlobals<128> g) {
    launch_triangle_outgoing<128>(g);
    cudaDeviceSynchronize();
}

void run_triangle_incoming_128(TriangleIncomingGlobals<128> g) {
    launch_triangle_incoming<128>(g);
    cudaDeviceSynchronize();
}

// PyBind11 module definition
PYBIND11_MODULE(triangle_mul_kernels, m) {
    m.doc() = "Genie-TK Triangle Multiplicative Update kernels";
    
    // Bind the outgoing kernel
    kittens::py::bind_function<run_triangle_outgoing_128>(
        m, "triangle_outgoing",
        &TriangleOutgoingGlobals<128>::a,
        &TriangleOutgoingGlobals<128>::b,
        &TriangleOutgoingGlobals<128>::out,
        &TriangleOutgoingGlobals<128>::mask
    );
    
    // Bind the incoming kernel
    kittens::py::bind_function<run_triangle_incoming_128>(
        m, "triangle_incoming",
        &TriangleIncomingGlobals<128>::a,
        &TriangleIncomingGlobals<128>::b,
        &TriangleIncomingGlobals<128>::out,
        &TriangleIncomingGlobals<128>::mask
    );
}
