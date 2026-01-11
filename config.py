# Genie-TK 配置
# 构建时需要的配置

# ThunderKittens 内核注册
sources = {
    'genie_triangle_mul': {
        'source_files': {
            'h100': 'kernels/triangle_mul/triangle_mul.cu',
            'a100': 'kernels/triangle_mul/triangle_mul.cu',
            '4090': 'kernels/triangle_mul/triangle_mul.cu',
        }
    },
    'genie_triangle_attention': {
        'source_files': {
            'h100': 'kernels/triangle_attention/triangle_attention.cu',
            'a100': 'kernels/triangle_attention/triangle_attention.cu',
            '4090': 'kernels/triangle_attention/triangle_attention.cu',
        }
    },
    'genie_layernorm': {
        'source_files': {
            'h100': 'kernels/fused_layernorm/layernorm.cu',
            'a100': 'kernels/fused_layernorm/layernorm.cu',
            '4090': 'kernels/fused_layernorm/layernorm.cu',
        }
    },
}

# 要构建的内核列表
kernels = [
    'genie_triangle_mul',
    'genie_triangle_attention',
    'genie_layernorm',
]

# GPU 目标 ('h100', 'a100', '4090')
target = 'h100'
