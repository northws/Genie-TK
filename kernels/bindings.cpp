/**
 * Genie-TK: Main C++ Bindings File
 * 
 * Combines all kernel modules into a single Python extension.
 * 
 * License: MIT
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declarations from kernel modules
namespace genie_tk {

namespace triangle_mul {
    torch::Tensor triangle_mul_outgoing_forward(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor mask
    );
    
    torch::Tensor triangle_mul_incoming_forward(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor mask
    );
}

namespace triangle_attention {
    torch::Tensor triangle_attention_starting_forward(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor bias,
        torch::Tensor mask,
        float scale
    );
    
    torch::Tensor triangle_attention_ending_forward(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor bias,
        torch::Tensor mask,
        float scale
    );
}

namespace layernorm {
    torch::Tensor layernorm_forward(
        torch::Tensor x,
        torch::Tensor gamma,
        torch::Tensor beta,
        float eps
    );
    
    torch::Tensor fused_layernorm_linear_forward(
        torch::Tensor x,
        torch::Tensor gamma,
        torch::Tensor beta,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        float eps
    );
}

} // namespace genie_tk

PYBIND11_MODULE(_C, m) {
    m.doc() = "Genie-TK: ThunderKittens-accelerated Triangle operations for protein structure prediction";
    
    // Triangle Multiplicative Update
    m.def("triangle_mul_outgoing", &genie_tk::triangle_mul::triangle_mul_outgoing_forward,
          "Triangle Multiplicative Update (Outgoing) - Algorithm 11",
          py::arg("a"), py::arg("b"), py::arg("mask"));
    
    m.def("triangle_mul_incoming", &genie_tk::triangle_mul::triangle_mul_incoming_forward,
          "Triangle Multiplicative Update (Incoming) - Algorithm 12",
          py::arg("a"), py::arg("b"), py::arg("mask"));
    
    // Triangle Attention
    m.def("triangle_attention_starting", &genie_tk::triangle_attention::triangle_attention_starting_forward,
          "Triangle Attention (Starting Node) - Algorithm 13",
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("bias"), py::arg("mask"), py::arg("scale"));
    
    m.def("triangle_attention_ending", &genie_tk::triangle_attention::triangle_attention_ending_forward,
          "Triangle Attention (Ending Node) - Algorithm 14",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("bias"), py::arg("mask"), py::arg("scale"));
    
    // LayerNorm
    m.def("layernorm", &genie_tk::layernorm::layernorm_forward,
          "LayerNorm forward",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    
    m.def("fused_layernorm_linear", &genie_tk::layernorm::fused_layernorm_linear_forward,
          "Fused LayerNorm + Linear forward",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("weight"),
          py::arg("bias") = c10::nullopt, py::arg("eps") = 1e-5f);
}
