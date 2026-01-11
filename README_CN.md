# Genie-TK: ThunderKittens 加速的蛋白质结构预测



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.3%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

## 概述

**Genie-TK** 是一种基于 [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 的高性能实现，用于 AlphaFold2/Genie 的三角操作。ThunderKittens 是一个用于编写快速、简单 GPU 内核的框架。

该库提供以下优化的 CUDA 内核：
- **三角乘法更新**（Outgoing 和 Incoming）- AlphaFold2 的算法 11 和 12
- **三角注意力**（起始节点和结束节点）- AlphaFold2 的算法 13 和 14
- **融合操作**，结合了 LayerNorm、投影和门控

---

## 主要特性

🚀 **高性能**
- 通过 ThunderKittens 原语优化张量核心利用率
- 使用双缓冲异步 TMA 加载以隐藏内存延迟
- 融合操作以最小化内存带宽需求

🧬 **蛋白质特定优化**
- 针对蛋白质建模中常见的对表示维度进行了优化
- 支持具有高效掩码的可变序列长度
- 内存高效的分解表示

⚡ **易于集成**
- 可直接替换 PyTorch 的三角操作
- 与现有的 AlphaFold2/OpenFold/Genie 实现兼容
- 提供干净的 Python API，并自动回退到 PyTorch

---

## 数学公式

本节提供了 AlphaFold2（Jumper 等，Nature 2021）中定义的三角操作的严格数学定义，以及我们的实现细节。

### 符号说明

| 符号 | 描述 |
|--------|-------------|
| $z_{ij} \in \mathbb{R}^{c_z}$ | 残基对 $(i, j)$ 的对表示 |
| $\mathbf{Z} \in \mathbb{R}^{N \times N \times c_z}$ | 完整的对表示张量 |
| $c_z$ | 对表示通道维度 |
| $c$ | 隐藏通道维度 |
| $H$ | 注意力头的数量 |
| $N$ | 序列长度 |
| $\sigma(\cdot)$ | Sigmoid 激活函数 |
| $\text{LN}(\cdot)$ | 层归一化 |

---

### 算法 11：三角乘法更新（Outgoing）

Outgoing 更新从残基 $i$ 和 $j$ 向它们的公共邻居 $k$ 传播信息，强制约束如果边 $(i, k)$ 和 $(j, k)$ 存在，则边 $(i, j)$ 应反映这种关系。

**数学定义：**

$$
\begin{aligned}
\bar{z}_{ij} &= \text{LayerNorm}(z_{ij}) \\
a_{ik} &= \sigma\left(W^{a,g} \bar{z}_{ik} + b^{a,g}\right) \odot \left(W^{a,p} \bar{z}_{ik} + b^{a,p}\right) \\
b_{jk} &= \sigma\left(W^{b,g} \bar{z}_{jk} + b^{b,g}\right) \odot \left(W^{b,p} \bar{z}_{jk} + b^{b,p}\right) \\
g_{ij} &= \sigma\left(W^{g} \bar{z}_{ij} + b^{g}\right) \\
z_{ij} &\leftarrow z_{ij} + g_{ij} \odot W^{z}\,\text{LayerNorm}\left(\sum_{k=1}^{N} a_{ik} \odot b_{jk}\right)
\end{aligned}
$$

---

## 安装

### 先决条件

- CUDA 12.3+（推荐 CUDA 12.6）
- 支持 C++20 的 GCC 11+
- Python 3.10+
- PyTorch 2.0+
- H100、A100 或 RTX 4090 GPU

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-org/genie-tk.git
cd genie-tk

# 设置环境
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens
source env.src

# 安装
pip install -e .
```

### 从源码安装

```bash
# 构建内核
cd Genie-TK
python setup.py install

# 运行测试
pytest tests/
```

## 快速开始

```python
import torch
from genie_tk import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
from genie_tk import TriangleAttentionStartingNode, TriangleAttentionEndingNode

# 创建对表示
batch_size, seq_len, hidden_dim = 1, 128, 128
z = torch.randn(batch_size, seq_len, seq_len, hidden_dim, device='cuda', dtype=torch.bfloat16)
mask = torch.ones(batch_size, seq_len, seq_len, device='cuda')

# 三角乘法更新（Outgoing）
tri_mul_out = TriangleMultiplicationOutgoing(c_z=hidden_dim, c_hidden=128).cuda()
z_updated = tri_mul_out(z, mask)

# 三角注意力（起始节点）
tri_att_start = TriangleAttentionStartingNode(c_in=hidden_dim, c_hidden=32, no_heads=4).cuda()
z_attended = tri_att_start(z, mask)
```

---

## 引用

```bibtex
@software{genie_tk2024,
  title={Genie-TK: ThunderKittens-Accelerated Protein Structure Prediction},
  author={Genie-TK Contributors},
  year={2024},
  url={https://github.com/your-org/genie-tk}
}
```

## 许可证

此项目基于 MIT 许可证 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- [HazyResearch](https://github.com/HazyResearch) 提供的 ThunderKittens
- DeepMind 提供的 AlphaFold2 架构设计
- OpenFold 团队提供的参考实现