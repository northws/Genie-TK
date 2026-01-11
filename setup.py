import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Project root
ROOT_DIR = Path(__file__).parent.absolute()
KERNELS_DIR = ROOT_DIR / "kernels"

# Get ThunderKittens root
THUNDERKITTENS_ROOT = os.getenv(
    'THUNDERKITTENS_ROOT',
    str(ROOT_DIR.parent / "packages" / "ThunderKittens")
)

# Detect GPU target
def detect_gpu_target():#Detect GPU architecture.
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        gpu_name = result.stdout.strip().lower()
        # Blackwell architecture - Data Center (B100, B200)
        if 'b200' in gpu_name or 'b100' in gpu_name:
            return 'b200', 'sm_100a'
        # Blackwell architecture - Consumer (RTX 50 series)
        elif '5090' in gpu_name or '5080' in gpu_name or '5070' in gpu_name:
            return 'b200', 'sm_100a'  # RTX 50系列使用相同的Blackwell架构
        elif 'h100' in gpu_name or 'h200' in gpu_name:
            return 'h100', 'sm_90a'
        elif 'a100' in gpu_name:
            return 'a100', 'sm_80'
        elif '4090' in gpu_name or '4080' in gpu_name or '4070' in gpu_name or '4060' in gpu_name:
            return '4090', 'sm_89'  # RTX 40系列 Ada Lovelace架构
        elif '3090' in gpu_name or '3080' in gpu_name or '3070' in gpu_name:
            return '3090', 'sm_86'  # RTX 30系列 Ampere架构
        else:
            print(f"Unknown GPU: {gpu_name}, defaulting to sm_80")
            return 'a100', 'sm_80'
    except Exception as e:
        print(f"Could not detect GPU: {e}, defaulting to sm_80")
        return 'a100', 'sm_80'

# Get Python and PyTorch include paths
def get_include_paths():
    python_include = subprocess.check_output([
        'python', '-c',
        "import sysconfig; print(sysconfig.get_path('include'))"
    ]).decode().strip()
    
    torch_includes = subprocess.check_output([
        'python', '-c',
        "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"
    ]).decode().strip()
    
    return python_include, torch_includes

TARGET, ARCH = detect_gpu_target()
PYTHON_INCLUDE, TORCH_INCLUDE = get_include_paths()

print(f"Building for target: {TARGET} ({ARCH})")
print(f"ThunderKittens root: {THUNDERKITTENS_ROOT}")

# CUDA compilation flags
CUDA_FLAGS = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-std=c++20',
    '-O3',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    f'-I{THUNDERKITTENS_ROOT}/include',
    f'-I{THUNDERKITTENS_ROOT}/prototype',
    f'-I{PYTHON_INCLUDE}',
    f'-I{KERNELS_DIR}',
    '-DTORCH_COMPILE',
    f'-arch={ARCH}',
]

# Add target-specific flags
if TARGET == 'b200':
    # Blackwell architecture (B100/B200, RTX 5090/5080/5070)
    # Requires both HOPPER and BLACKWELL flags
    CUDA_FLAGS.append('-DKITTENS_HOPPER')
    CUDA_FLAGS.append('-DKITTENS_BLACKWELL')
elif TARGET == 'h100':
    CUDA_FLAGS.append('-DKITTENS_HOPPER')
elif TARGET == 'a100':
    CUDA_FLAGS.append('-DKITTENS_A100')
elif TARGET == '4090':
    CUDA_FLAGS.append('-DKITTENS_4090')  # Ada Lovelace
elif TARGET == '3090':
    CUDA_FLAGS.append('-DKITTENS_AMPERE')  # Ampere consumer

# Add torch includes
CUDA_FLAGS.extend(TORCH_INCLUDE.split())

CPP_FLAGS = ['-std=c++20', '-O3']

# Source files
KERNEL_SOURCES = [
    str(KERNELS_DIR / "triangle_mul" / "triangle_mul.cu"),
    str(KERNELS_DIR / "triangle_attention" / "triangle_attention.cu"),
    str(KERNELS_DIR / "fused_layernorm" / "layernorm.cu"),
]

# Filter existing sources
KERNEL_SOURCES = [s for s in KERNEL_SOURCES if os.path.exists(s)]

# Add Python binding file
BINDING_SOURCE = str(KERNELS_DIR / "bindings.cpp")
if os.path.exists(BINDING_SOURCE):
    KERNEL_SOURCES.append(BINDING_SOURCE)

if not KERNEL_SOURCES:
    print("Warning: No kernel source files found. Building Python-only package.")
    ext_modules = []
else:
    ext_modules = [
        CUDAExtension(
            name='genie_tk._C',
            sources=KERNEL_SOURCES,
            extra_compile_args={
                'cxx': CPP_FLAGS,
                'nvcc': CUDA_FLAGS,
            },
            libraries=['cuda'],
        )
    ]

setup(
    name='genie-tk',
    version='0.1.0',
    description='ThunderKittens-accelerated Triangle operations for protein structure prediction',
    author='Genie-TK Contributors',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'einops>=0.6.0',
    ],
)
