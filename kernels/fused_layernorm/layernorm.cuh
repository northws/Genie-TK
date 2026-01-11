/**
 * Genie-TK: Fused LayerNorm Kernels
 * 
 * Optimized LayerNorm operations for pair representations.
 * Supports fused LayerNorm + linear projection for memory efficiency.
 * 
 * License: MIT
 */

#pragma once

#include "kittens.cuh"

namespace genie_tk {
namespace layernorm {

using namespace kittens;

// Configuration
constexpr int BLOCK_SIZE = 256;
constexpr float EPS = 1e-5f;

// ============================================================================
// Warp-level LayerNorm for small dimensions
// ============================================================================

template<int D>
__device__ __forceinline__ void warp_layernorm(
    float* __restrict__ x,      // [D] input/output
    const float* __restrict__ gamma,  // [D] scale
    const float* __restrict__ beta,   // [D] bias
    float eps = EPS
) {
    const int lane = threadIdx.x % 32;
    
    // Compute mean
    float sum = 0.0f;
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        sum += x[i];
    }
    sum = warpReduceSum(sum);
    float mean = sum / D;
    
    // Compute variance
    float var_sum = 0.0f;
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warpReduceSum(var_sum);
    float rstd = rsqrtf(var_sum / D + eps);
    
    // Normalize and apply affine
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        x[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

// ============================================================================
// Block-level LayerNorm for larger dimensions
// ============================================================================

template<int D>
__global__ void layernorm_kernel(
    float* __restrict__ x,           // [N, D]
    const float* __restrict__ gamma, // [D]
    const float* __restrict__ beta,  // [D]
    int N,
    float eps = EPS
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    
    float* row_x = x + row * D;
    
    extern __shared__ float smem[];
    float* s_gamma = smem;
    float* s_beta = smem + D;
    
    // Load gamma and beta to shared memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        s_gamma[i] = gamma[i];
        s_beta[i] = beta[i];
    }
    __syncthreads();
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += row_x[i];
    }
    
    __shared__ float s_mean, s_rstd;
    
    // Block reduce for mean
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        s_mean = sum / D;
    }
    __syncthreads();
    
    float mean = s_mean;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = row_x[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = blockReduceSum(var_sum);
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(var_sum / D + eps);
    }
    __syncthreads();
    
    float rstd = s_rstd;
    
    // Normalize and apply affine
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_x[i] = (row_x[i] - mean) * rstd * s_gamma[i] + s_beta[i];
    }
}

// ============================================================================
// Fused LayerNorm + Linear Projection
// Useful for the input normalization in Triangle operations
// ============================================================================

template<int D_IN, int D_OUT>
__global__ void fused_layernorm_linear_kernel(
    const float* __restrict__ x,           // [N, D_IN]
    const float* __restrict__ gamma,       // [D_IN]
    const float* __restrict__ beta,        // [D_IN]
    const float* __restrict__ weight,      // [D_IN, D_OUT]
    const float* __restrict__ bias_out,    // [D_OUT] (optional)
    float* __restrict__ out,               // [N, D_OUT]
    int N,
    float eps = EPS
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    
    const float* row_x = x + row * D_IN;
    float* row_out = out + row * D_OUT;
    
    extern __shared__ float smem[];
    float* s_normalized = smem;  // [D_IN]
    
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
        sum += row_x[i];
    }
    
    __shared__ float s_mean, s_rstd;
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) s_mean = sum / D_IN;
    __syncthreads();
    
    // Step 2: Compute variance and normalize
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
        float diff = row_x[i] - s_mean;
        var_sum += diff * diff;
    }
    
    var_sum = blockReduceSum(var_sum);
    if (threadIdx.x == 0) s_rstd = rsqrtf(var_sum / D_IN + eps);
    __syncthreads();
    
    // Step 3: Store normalized values to shared memory
    for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
        s_normalized[i] = (row_x[i] - s_mean) * s_rstd * gamma[i] + beta[i];
    }
    __syncthreads();
    
    // Step 4: Linear projection
    for (int j = threadIdx.x; j < D_OUT; j += blockDim.x) {
        float acc = (bias_out != nullptr) ? bias_out[j] : 0.0f;
        for (int i = 0; i < D_IN; i++) {
            acc += s_normalized[i] * weight[i * D_OUT + j];
        }
        row_out[j] = acc;
    }
}

// Helper functions for block reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

} // namespace layernorm
} // namespace genie_tk
