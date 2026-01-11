# 安装指南

## 系统要求

- CUDA 12.3+ (推荐 CUDA 12.6)
- GCC 11+ (支持 C++20)
- Python 3.10+
- PyTorch 2.0+
- H100 / A100 / RTX 4090 GPU

## 快速安装

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/genie-tk.git
cd genie-tk
```

### 2. 设置环境

```bash
# 设置 ThunderKittens 路径
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens

# 加载环境变量
source env.src
```

### 3. 安装

```bash
# 安装 Python 包（包含 CUDA 内核编译）
pip install -e .
```

## 仅安装 Python 包（无 CUDA 内核）

如果你不需要 CUDA 加速或没有兼容的 GPU:

```bash
pip install -e . --no-build-isolation
```

这将安装纯 PyTorch 实现，不编译 CUDA 内核。

## 验证安装

```python
import genie_tk

# 检查 CUDA 内核是否可用
print(f"CUDA kernels available: {genie_tk.has_cuda_kernels()}")

# 查看设备信息
print(genie_tk.get_device_info())
```

## 常见问题

### CUDA 版本不匹配

确保你的 CUDA 版本与 PyTorch 编译时使用的版本匹配:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### GCC 版本过低

ThunderKittens 需要 C++20 支持:

```bash
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
```

### 找不到 nvcc

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```
