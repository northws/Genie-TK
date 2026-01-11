# 安装指南

## 系统要求

- CUDA 12.3+ (推荐 CUDA 12.6，Blackwell 架构需要 CUDA 12.8+)
- GCC 11+ (支持 C++20)
- Python 3.10+
- PyTorch 2.0+
- 支持的 GPU:
  - **Blackwell (数据中心)**: B100, B200 (sm_100a) - 需要 ThunderKittens 3.0
  - **Blackwell (消费级)**: RTX 5090, RTX 5080, RTX 5070 (sm_100a)
  - **Hopper**: H100, H200 (sm_90a)
  - **Ampere (数据中心)**: A100 (sm_80)
  - **Ada Lovelace**: RTX 4090, RTX 4080, RTX 4070, RTX 4060 (sm_89)
  - **Ampere (消费级)**: RTX 3090, RTX 3080, RTX 3070 (sm_86)

## ThunderKittens 3.0 更新

Genie-TK 现已支持 ThunderKittens 3.0，主要变更包括：
- 支持 Blackwell 架构 (B100/B200) GPU
- Warp scope 现在需要显式声明为 `kittens::warp::`
- 代码重构和 Megakernels 支持

## 快速安装

### 1. 克隆仓库

```bash
git clone https://github.com/northws/genie-tk.git
cd genie-tk
```

### 2. 设置环境

#### Linux / macOS
```bash
# 设置 ThunderKittens 路径
git clone https://github.com/HazyResearch/ThunderKittens.git
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

### GPU 架构选择

Genie-TK 会自动检测 GPU 并选择正确的编译目标：

| GPU | 编译标志 | 架构 |
|-----|---------|------|
| B200/B100 | `-DKITTENS_HOPPER -DKITTENS_BLACKWELL` | sm_100a |
| RTX 5090/5080/5070 | `-DKITTENS_HOPPER -DKITTENS_BLACKWELL` | sm_100a |
| H100/H200 | `-DKITTENS_HOPPER` | sm_90a |
| A100 | `-DKITTENS_A100` | sm_80 |
| RTX 4090/4080/4070/4060 | `-DKITTENS_4090` | sm_89 |
| RTX 3090/3080/3070 | `-DKITTENS_AMPERE` | sm_86 |

手动指定目标架构：
```python
# 编辑 config.py
target = 'b200'  # 或 'h100', 'a100', '4090'
```

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
