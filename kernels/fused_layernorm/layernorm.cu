/**
 * Genie-TK: LayerNorm CUDA Implementation
 * 
 * License: MIT
 */

#include "layernorm.cuh"
#include <torch/extension.h>

namespace genie_tk {
namespace layernorm {

// ============================================================================
// PyTorch Interface Functions
// ============================================================================

torch::Tensor layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA tensor");
    
    const int N = x.numel() / x.size(-1);
    const int D = x.size(-1);
    
    auto out = x.clone();
    
    int block_size = std::min(BLOCK_SIZE, D);
    size_t smem_size = 2 * D * sizeof(float);
    
    layernorm_kernel<128><<<N, block_size, smem_size>>>(
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N,
        eps
    );
    
    return out;
}

torch::Tensor fused_layernorm_linear_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    
    const int N = x.numel() / x.size(-1);
    const int D_IN = x.size(-1);
    const int D_OUT = weight.size(1);
    
    auto out = torch::empty({x.size(0), D_OUT}, x.options());
    
    int block_size = std::min(BLOCK_SIZE, std::max(D_IN, D_OUT));
    size_t smem_size = D_IN * sizeof(float);
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    fused_layernorm_linear_kernel<128, 128><<<N, block_size, smem_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        N,
        eps
    );
    
    return out;
}

} // namespace layernorm
} // namespace genie_tk

// ============================================================================
// PyBind11 Module
// ============================================================================

PYBIND11_MODULE(layernorm_kernels, m) {
    m.doc() = "Genie-TK LayerNorm kernels";
    
    m.def("layernorm_forward", &genie_tk::layernorm::layernorm_forward,
          "LayerNorm forward pass",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    
    m.def("fused_layernorm_linear_forward", &genie_tk::layernorm::fused_layernorm_linear_forward,
          "Fused LayerNorm + Linear forward pass",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("weight"),
          py::arg("bias") = c10::nullopt, py::arg("eps") = 1e-5f);
}
