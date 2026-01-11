# Genie-TK 文档

## 概述

Genie-TK 是一个使用 ThunderKittens 加速的蛋白质结构预测库，实现了 AlphaFold2 中的 Triangle 操作。

## 目录

- [安装](installation.md)
- [快速入门](quickstart.md)
- [API 参考](api/index.md)
- [性能基准](benchmarks.md)

## 核心概念

### Triangle Multiplicative Update

Triangle Multiplicative Update 是 AlphaFold2 中用于更新 pair representation 的关键操作。它通过沿三角形边传播信息来强化几何一致性。

**Outgoing (算法 11)**:
```
Z_ij += Σ_k (a_ik ⊗ b_jk)
```
信息从残基 i 和 j 流向它们的共同邻居 k。

**Incoming (算法 12)**:
```
Z_ij += Σ_k (a_ki ⊗ b_kj)
```
信息从共同邻居 k 流向残基 i 和 j。

### Triangle Attention

Triangle Attention 在 pair representation 上计算带有三角形偏置项的自注意力。

**Starting Node (算法 13)**: 沿行计算注意力
**Ending Node (算法 14)**: 沿列计算注意力

## ThunderKittens 优化

Genie-TK 使用 ThunderKittens 提供的以下优化:

1. **Tensor Core 利用**: 通过 TK 的 mma 原语实现高效矩阵乘法
2. **异步 TMA 加载**: 双缓冲隐藏内存延迟
3. **融合操作**: 减少内存带宽需求
4. **Tile-based 计算**: 优化共享内存使用

## 参考文献

1. Spector, B., et al. "ThunderKittens: Simple, Fast, and Cute GPU Kernels." HazyResearch, 2024.
2. Jumper, J., et al. "Highly accurate protein structure prediction with AlphaFold." Nature, 2021.
3. Lin, Y., et al. "Genie: De novo protein design by equivariantly diffusing oriented residue clouds." arXiv:2301.12485, 2023.
