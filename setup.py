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
        if 'h100' in gpu_name or 'h200' in gpu_name:
            return 'h100', 'sm_90a'
        elif 'a100' in gpu_name:
            return 'a100', 'sm_80'
        elif '4090' in gpu_name or '4080' in gpu_name or '3090' in gpu_name:
            return '4090', 'sm_89'
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
if TARGET == 'h100':
    CUDA_FLAGS.append('-DKITTENS_HOPPER')
elif TARGET == 'a100':
    CUDA_FLAGS.append('-DKITTENS_A100')
elif TARGET == '4090':
    CUDA_FLAGS.append('-DKITTENS_4090')

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
