# Genie-TK 配置
# 构建时需要的配置

# ThunderKittens 内核注册
sources = {
    'genie_triangle_mul': {
        'source_files': {
            'h100': 'kernels/triangle_mul/triangle_mul.cu',
            'a100': 'kernels/triangle_mul/triangle_mul.cu',
            '4090': 'kernels/triangle_mul/triangle_mul.cu',
            '3090': 'kernels/triangle_mul/triangle_mul.cu',  # Ampere consumer
            'b200': 'kernels/triangle_mul/triangle_mul.cu',  # Blackwell
        }
    },
    'genie_triangle_attention': {
        'source_files': {
            'h100': 'kernels/triangle_attention/triangle_attention.cu',
            'a100': 'kernels/triangle_attention/triangle_attention.cu',
            '4090': 'kernels/triangle_attention/triangle_attention.cu',
            '3090': 'kernels/triangle_attention/triangle_attention.cu',  # Ampere consumer
            'b200': 'kernels/triangle_attention/triangle_attention.cu',  # Blackwell
        }
    },
    'genie_layernorm': {
        'source_files': {
            'h100': 'kernels/fused_layernorm/layernorm.cu',
            'a100': 'kernels/fused_layernorm/layernorm.cu',
            '4090': 'kernels/fused_layernorm/layernorm.cu',
            '3090': 'kernels/fused_layernorm/layernorm.cu',  # Ampere consumer
            'b200': 'kernels/fused_layernorm/layernorm.cu',  # Blackwell
        }
    },
}

# 要构建的内核列表
kernels = [
    'genie_triangle_mul',
    'genie_triangle_attention',
    'genie_layernorm',
]

# GPU 目标 ('h100', 'a100', '4090', '3090', 'b200')
# B200 为 Blackwell 架构 GPU (包括 RTX 50 系列)
# 4090 为 Ada Lovelace 架构 (RTX 40 系列)
# 3090 为 Ampere 消费级架构 (RTX 30 系列)
target = 'h100'
