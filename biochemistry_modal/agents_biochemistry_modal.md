# agents_biochemistry_modal.md

## 任务背景

这是华山项目的**生化模态编码模块**。

当前整体任务为：

```text
T0 CT + T0 生化 + Age/Sex + future time
    ↓
多模态融合
    ↓
预测未来 0–2 年生化指标
```

目前数据模块已经能够提供：

```python
{
    "bio_base": torch.FloatTensor,      # [10]
    "base_mask": torch.FloatTensor,     # [10]
    "demographics": torch.FloatTensor,  # [2]
    "time": torch.FloatTensor,          # [1]
    "target": torch.FloatTensor,        # [10]
    "target_mask": torch.FloatTensor,   # [10]
}
```

其中 `bio_base` 已经由 `multidataset/huashan_dataset.py` 完成 train-only normalization、CysC 缺失值 train-median 填补，并保证不含 NaN/Inf。

本模块只负责：

\[
Bio_{T0} + MissingMask \rightarrow z_{Bio}
\]

不要重复做 normalization。

## 1. 项目结构要求

请在当前项目根目录下创建：

```text
biochemistry_modal/
└── biochemistry_encoder.py
```

不要创建任何二级代码目录，例如 `utils/`、`configs/`、`models/`、`tests/`。不要使用 YAML、Hydra、registry、factory 或复杂继承体系。

## 2. 固定指标顺序

任何地方都不要改变：

```python
BIOMARKERS = [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
    "CysC",
]
```

因此：

```text
bio_base.shape[-1] == 10
base_mask.shape[-1] == 10
```

Mask 约定固定：

```text
1 = observed
0 = missing
```

目前前 9 个核心指标 mask 应为 1，第 10 个 CysC 允许 0/1。

## 3. Encoder 输入

第一版直接 concat：

```python
x = torch.cat([bio_base, base_mask], dim=-1)
```

得到 20 维输入：

\[
z_{Bio}=MLP([Bio_{T0},Mask])
\]

虽然当前只有 CysC 的 mask 会变化，但保留完整 10 维 mask 可以保持接口统一，并方便以后其他指标出现 missing。

## 4. 默认网络结构

保持简单：

```text
20
 ↓
Linear(20, 64)
 ↓
LayerNorm(64)
 ↓
GELU
 ↓
Dropout(0.10)
 ↓
Linear(64, 128)
 ↓
LayerNorm(128)
 ↓
GELU
 ↓
z_bio = 128
```

默认参数：

```python
BiochemistryEncoder(
    input_dim=10,
    hidden_dim=64,
    embedding_dim=128,
    dropout=0.1,
)
```

其中 `input_dim=10` 表示生化指标数量，真正第一层输入维度应为 `input_dim * 2`，因为还包含 mask。

默认使用 LayerNorm，不使用 BatchNorm。

## 5. Forward API

推荐：

```python
def forward(
    self,
    bio_base: torch.Tensor,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

支持单样本：

```text
bio_base: [10]
base_mask: [10]
output: [128]
```

支持 batch：

```text
bio_base: [B,10]
base_mask: [B,10]
output: [B,128]
```

不要让该类直接接收整个 Dataset dict。

## 6. 输入检查

至少检查：

- `bio_base` 与 `base_mask` shape 一致
- 最后一维必须等于 10
- 两者均为 finite
- mask 只能包含 0/1

异常时抛出可读的 `ValueError`，不要静默修正。

## 7. 不要重复处理 CysC

Dataset 层已经完成：

```text
CysC missing
    ↓
train median fill
    ↓
base_mask[9] = 0
```

因此禁止：

- 重新计算 CysC median
- 再次 normalization
- 在 encoder 内做缺失填补

## 8. 模块边界

本模块只处理 T0 生化。

不要处理：

- age / sex
- time
- CT
- target
- prediction head
- loss
- fusion
- decoder
- Transformer
- training loop

## 9. 主要类

只需要：

```python
class BiochemistryEncoder(nn.Module):
    ...
```

可以有少量 helper，但不要过度封装。

## 10. CLI smoke test

`biochemistry_encoder.py` 本身支持：

```bash
python biochemistry_modal/biochemistry_encoder.py --smoke-test
```

测试：

```python
torch.manual_seed(42)

bio = torch.randn(4, 10)
mask = torch.ones(4, 10)
mask[0, 9] = 0
mask[2, 9] = 0
```

检查：

```text
input bio shape:        [4,10]
input mask shape:       [4,10]
output embedding shape: [4,128]
```

断言：

```python
assert output.shape == (4, 128)
assert torch.isfinite(output).all()
```

再测试单样本：

```text
[10] + [10] -> [128]
```

Smoke test 同时打印 trainable parameter count。

## 11. 依赖

尽量只使用：

```text
torch
argparse
```

## 12. 验收标准

- [ ] 创建 `biochemistry_modal/`
- [ ] 不创建代码子目录
- [ ] 存在 `biochemistry_encoder.py`
- [ ] 指标顺序固定为 10 项
- [ ] bio + mask concat 后进入 MLP
- [ ] 默认 hidden_dim=64
- [ ] 默认 embedding_dim=128
- [ ] 默认 dropout=0.1
- [ ] 默认 LayerNorm
- [ ] 支持 `[10] -> [128]`
- [ ] 支持 `[B,10] -> [B,128]`
- [ ] mask 只允许 0/1
- [ ] 输出无 NaN/Inf
- [ ] 不重新 normalization
- [ ] 不重新填补 CysC
- [ ] 不实现预测与训练逻辑
- [ ] CLI smoke test 通过

## 13. 完成后请汇报

请告诉我：

1. 创建了哪些文件
2. `BiochemistryEncoder` 的实际结构
3. 参数量
4. batch 与 single-sample smoke test shape
5. mask 检查是否正常
6. 是否发现与现有 Dataset 接口不一致
7. 下一步如何 import，例如：

```python
from biochemistry_modal.biochemistry_encoder import BiochemistryEncoder
```

如果现有 `multidataset/huashan_dataset.py` 的输出接口与本文不一致，请优先说明，不要擅自修改已有 Dataset。
