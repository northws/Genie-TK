/**
 * Genie-TK: Triangle Attention CUDA Implementation
 * 
 * License: MIT
 */

#include "triangle_attention.cuh"
#include "pyutils/pyutils.cuh"

namespace genie_tk {
namespace triangle_attention {

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

template<int C_IN, int C_HIDDEN, int N_HEADS>
void launch_triangle_attention_starting(
    TriangleAttentionStartingGlobals<C_IN, C_HIDDEN, N_HEADS>& g,
    cudaStream_t stream = 0
) {
    size_t smem_size = sizeof(q_smem_tile) + 
                       sizeof(k_smem_tile) * STAGES + 
                       sizeof(v_smem_tile) * STAGES +
                       sizeof(o_smem_tile);
    
    triangle_attention_starting_kernel<C_IN, C_HIDDEN, N_HEADS>
        <<<g.grid(), g.block(), smem_size, stream>>>(g);
}

template<int C_IN, int C_HIDDEN, int N_HEADS>
void launch_triangle_attention_ending(
    TriangleAttentionEndingGlobals<C_IN, C_HIDDEN, N_HEADS>& g,
    cudaStream_t stream = 0
) {
    size_t smem_size = sizeof(q_smem_tile) + 
                       sizeof(k_smem_tile) * STAGES + 
                       sizeof(v_smem_tile) * STAGES +
                       sizeof(o_smem_tile);
    
    triangle_attention_ending_kernel<C_IN, C_HIDDEN, N_HEADS>
        <<<g.grid(), g.block(), smem_size, stream>>>(g);
}

// Explicit instantiations
template void launch_triangle_attention_starting<128, 32, 4>(
    TriangleAttentionStartingGlobals<128, 32, 4>&, cudaStream_t);
template void launch_triangle_attention_ending<128, 32, 4>(
    TriangleAttentionEndingGlobals<128, 32, 4>&, cudaStream_t);

} // namespace triangle_attention
} // namespace genie_tk

// ============================================================================
// Python Bindings
// ============================================================================

using namespace genie_tk::triangle_attention;

void run_triangle_attention_starting(TriangleAttentionStartingGlobals<128, 32, 4> g) {
    launch_triangle_attention_starting<128, 32, 4>(g);
    cudaDeviceSynchronize();
}

void run_triangle_attention_ending(TriangleAttentionEndingGlobals<128, 32, 4> g) {
    launch_triangle_attention_ending<128, 32, 4>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(triangle_attention_kernels, m) {
    m.doc() = "Genie-TK Triangle Attention kernels";
    
    kittens::py::bind_function<run_triangle_attention_starting>(
        m, "triangle_attention_starting",
        &TriangleAttentionStartingGlobals<128, 32, 4>::q,
        &TriangleAttentionStartingGlobals<128, 32, 4>::k,
        &TriangleAttentionStartingGlobals<128, 32, 4>::v,
        &TriangleAttentionStartingGlobals<128, 32, 4>::out
    );
    
    kittens::py::bind_function<run_triangle_attention_ending>(
        m, "triangle_attention_ending",
        &TriangleAttentionEndingGlobals<128, 32, 4>::q,
        &TriangleAttentionEndingGlobals<128, 32, 4>::k,
        &TriangleAttentionEndingGlobals<128, 32, 4>::v,
        &TriangleAttentionEndingGlobals<128, 32, 4>::out
    );
}
