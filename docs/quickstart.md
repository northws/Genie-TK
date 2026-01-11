# 快速入门

## 基本用法

### Triangle Multiplicative Update

```python
import torch
from genie_tk import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming

# 创建 pair representation
batch_size = 1
seq_len = 128
c_z = 128  # pair representation 维度

z = torch.randn(batch_size, seq_len, seq_len, c_z, device='cuda')
mask = torch.ones(batch_size, seq_len, seq_len, device='cuda')

# Triangle Multiplicative Update (Outgoing)
tri_mul_out = TriangleMultiplicationOutgoing(
    c_z=c_z,
    c_hidden=128,
).cuda()

z_updated = z + tri_mul_out(z, mask)  # 残差连接

# Triangle Multiplicative Update (Incoming)
tri_mul_in = TriangleMultiplicationIncoming(
    c_z=c_z,
    c_hidden=128,
).cuda()

z_updated = z_updated + tri_mul_in(z_updated, mask)
```

### Triangle Attention

```python
from genie_tk import TriangleAttentionStartingNode, TriangleAttentionEndingNode

# Triangle Attention (Starting Node)
tri_att_start = TriangleAttentionStartingNode(
    c_in=c_z,
    c_hidden=32,
    no_heads=4,
).cuda()

z_attended = z + tri_att_start(z, mask)

# Triangle Attention (Ending Node)
tri_att_end = TriangleAttentionEndingNode(
    c_in=c_z,
    c_hidden=32,
    no_heads=4,
).cuda()

z_attended = z_attended + tri_att_end(z_attended, mask)
```

### 完整的 Pair Transform

```python
from genie_tk import PairTransformNet

# 创建完整的 Pair Transform 网络
pair_net = PairTransformNet(
    c_p=128,
    n_layers=4,
    include_mul_update=True,
    include_tri_att=True,
    c_hidden_mul=128,
    c_hidden_tri_att=32,
    n_head_tri=4,
    tri_dropout=0.25,
).cuda()

# 前向传播
z_transformed = pair_net(z, mask)
```

## 与现有代码集成

Genie-TK 模块设计为 OpenFold/AlphaFold2 实现的直接替换:

```python
# 原来的 OpenFold 代码
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing as OFTriMulOut,
)

# 替换为 Genie-TK
from genie_tk import TriangleMultiplicationOutgoing as TKTriMulOut

# 使用相同的参数
layer = TKTriMulOut(c_z=128, c_hidden=128)
```

## 禁用 CUDA 内核

如果需要使用纯 PyTorch 实现（例如用于调试）:

```python
layer = TriangleMultiplicationOutgoing(
    c_z=128,
    c_hidden=128,
    use_fused_kernel=False,  # 使用 PyTorch 实现
)
```

## 性能提示

1. **使用 BF16**: CUDA 内核针对 BF16 优化
2. **批处理**: 尽可能批处理多个序列
3. **序列长度**: 序列长度为 64 的倍数时性能最佳
4. **预热**: 第一次调用会有编译开销
