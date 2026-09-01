# agents_fusion_query.md

## 任务背景

这是华山项目的**多模态融合 + MLP Time-Query 模块**。

当前目标：

\[
CT_{T0} + Bio_{T0} + Demographics + t \rightarrow Bio(t)
\]

上游预计提供：

```text
CT encoder:
    z_CT  -> [128]

BiochemistryEncoder:
    z_Bio -> [128]

Dataset:
    demographics -> [2]
    time         -> [1]
    target       -> [10]
    target_mask  -> [10]
```

本模块只负责：

```text
z_CT
  +
z_Bio
  +
Age/Sex
  ↓
Fusion
  ↓
baseline latent state z0
  +
future time
  ↓
MLP Time-Query Decoder
  ↓
predicted future biochemical vector [10]
```

第一版明确采用：

```text
Concat Fusion + MLP Time Embedding + MLP Decoder
```

不要第一版就使用 Transformer / Cross-Attention。

## 1. 项目结构要求

请在当前项目根目录创建：

```text
fusion_query_modal/
└── fusion_time_query.py
```

不要创建任何代码子目录。不要使用 Hydra、YAML、registry、factory 或复杂模型框架。

## 2. 非常重要的上游假设

本模块**不负责 CT 图像编码**。

它接收的是：

```text
ct_embedding
```

而不是 `[3,224,224]` CT image。

默认：

```text
ct_embedding_dim = 128
bio_embedding_dim = 128
```

### 必须检查 CT encoder 是否已经存在

之前 `CT_modal/` 的设计主要完成：

```text
DICOM
→ NIfTI
→ L2/L3/L4
→ ROI
→ 2.5D npy
```

如果当前项目还没有：

```text
2.5D ResNet -> z_CT
```

那么整个 end-to-end pipeline 仍然缺一个 CT encoder。

本任务**不要擅自把 ResNet 塞进 fusion_query_modal**。请在完成报告中明确说明 CT encoder 是否已经存在，以及实际输出维度。

## 3. Forward 输入

默认 batch shape：

```text
ct_embedding: [B,128]
bio_embedding:[B,128]
demographics: [B,2]
time:         [B,1]
```

也应支持单样本：

```text
[128], [128], [2], [1]
```

## 4. Demographics encoder

`demographics` 顺序固定：

```text
[
    normalized_age,
    sex_code
]
```

其中 age 已由 Dataset 使用 train-only mean/std 标准化，sex_code=0/1。

不要再次 normalization。

默认：

```text
2
 ↓
Linear(2,16)
 ↓
GELU
 ↓
Linear(16,32)
 ↓
GELU
 ↓
z_demo [32]
```

## 5. Fusion

默认：

```text
z_CT   = 128
z_Bio  = 128
z_Demo = 32
```

concat：

```text
128 + 128 + 32 = 288
```

然后：

```text
288
 ↓
Linear(288,128)
 ↓
LayerNorm(128)
 ↓
GELU
 ↓
Dropout(0.10)
 ↓
Linear(128,128)
 ↓
GELU
 ↓
z0 [128]
```

`z0` 表示患者 T0 multimodal baseline latent state。

第一版坚持 concat fusion，不实现 attention / gated fusion / tensor fusion / bilinear fusion。

## 6. Time 输入

Dataset 已经提供：

```text
time = t_days / 365.25
```

因此本模块不要再次除以 365.25。

## 7. Time Encoder

第一版：

```text
1
 ↓
Linear(1,32)
 ↓
GELU
 ↓
Linear(32,32)
 ↓
GELU
 ↓
e_t [32]
```

这就是当前的 MLP Time Query。

## 8. Time-conditioned Decoder

concat：

```text
z0 [128] + e_t [32] = [160]
```

decoder：

```text
160
 ↓
Linear(160,128)
 ↓
LayerNorm(128)
 ↓
GELU
 ↓
Dropout(0.10)
 ↓
Linear(128,64)
 ↓
GELU
 ↓
Linear(64,10)
 ↓
prediction [10]
```

数学形式：

\[
z_0=Fusion(z_{CT},z_{Bio},z_{demo})
\]

\[
e_t=TimeMLP(t)
\]

\[
\hat y(t)=Decoder([z_0,e_t])
\]

## 9. 输出顺序

固定：

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

输出：

```text
[B,10]
```

顺序与 Dataset target 完全一致。

## 10. 第一版预测定义

第一版只预测：

```text
standardized absolute target
```

暂时不要实现：

- delta prediction
- relative change
- percentage change

原因是当前 baseline 与 target 使用各自 train-only statistics，delta 预测若处理不严谨容易造成尺度混乱。

## 11. 推荐主要类

```python
class FusionTimeQueryHead(nn.Module):
    ...
```

Forward：

```python
def forward(
    self,
    ct_embedding: torch.Tensor,
    bio_embedding: torch.Tensor,
    demographics: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    ...
```

默认返回 prediction。

可以可选支持：

```python
return_features=False
```

若为 True，可返回：

```python
{
    "prediction": prediction,
    "baseline_state": z0,
    "time_embedding": e_t,
    "demo_embedding": z_demo,
}
```

如果增加明显复杂度则不要实现。

## 12. 输入检查

必须检查：

- ct 最后一维等于 `ct_embedding_dim`
- bio 最后一维等于 `bio_embedding_dim`
- demographics 最后一维等于 2
- time 最后一维等于 1
- batch 维一致
- 所有输入 finite

异常时抛出可读 `ValueError`。

## 13. Mask 的职责

`base_mask` 已经在 BiochemistryEncoder 中参与编码。

`target_mask` 只用于 loss。

因此 `FusionTimeQueryHead.forward()` 不接收 `base_mask` 或 `target_mask`。

## 14. Masked Huber Loss helper

在同一个文件中提供一个简单 helper：

```python
def masked_huber_loss(
    prediction,
    target,
    target_mask,
    delta=1.0,
):
    ...
```

输入：

```text
prediction  [B,10]
target      [B,10]
target_mask [B,10]
```

约定：

```text
1 = supervised
0 = missing
```

逻辑：

```text
elementwise Huber loss
× target_mask
÷ valid mask count
```

需要安全检查 `target_mask.sum() > 0`。

不要为 10 个指标分别写 10 个 loss。

## 15. 不要 inverse normalization

模型输出保持 standardized target space。

本模块不要读取 `normalization.json`，也不要计算临床单位 MAE。

## 16. Ablation 兼容性

后续需要比较：

```text
Bio only
Bio + Demo
CT + Bio
CT + Bio + Demo
```

当前不要为此构建复杂框架。

如果实现开关，只允许简单：

```python
use_ct=True
use_bio=True
use_demo=True
```

否则由外部传入 `zeros_like()` 也可以。

## 17. 模块解耦

`fusion_time_query.py`：

- 不 import Dataset
- 不读 Excel
- 不直接 import BiochemistryEncoder
- 不实现 CT encoder

保持 tensor in / tensor out。

未来总模型组合：

```python
ct_embedding = ct_encoder(ct)
bio_embedding = bio_encoder(bio_base, base_mask)

prediction = fusion_query(
    ct_embedding=ct_embedding,
    bio_embedding=bio_embedding,
    demographics=demographics,
    time=time,
)
```

## 18. Smoke test

支持：

```bash
python fusion_query_modal/fusion_time_query.py --smoke-test
```

测试：

```python
torch.manual_seed(42)

B = 4
ct = torch.randn(B,128)
bio = torch.randn(B,128)
demo = torch.randn(B,2)
time = torch.rand(B,1) * 2.0
```

断言：

```python
prediction.shape == (4,10)
torch.isfinite(prediction).all()
```

再测试单样本：

```text
[128],[128],[2],[1] -> [10]
```

同时打印 trainable parameter count。

## 19. Loss smoke test

构造：

```python
target = torch.randn(4,10)
mask = torch.ones(4,10)
mask[0,9] = 0
mask[2,9] = 0
```

检查：

- loss 是 scalar
- loss finite

再做一个 correctness test：

1. 记录原 loss
2. 将 `mask=0` 的 CysC target 改成非常大的极端值
3. 重新计算 loss
4. 两次 loss 应基本一致

证明缺失 CysC 不参与监督。

## 20. 默认参数

建议：

```python
ct_embedding_dim = 128
bio_embedding_dim = 128

demo_hidden_dim = 16
demo_embedding_dim = 32

fusion_hidden_dim = 128
fusion_dim = 128

time_hidden_dim = 32
time_embedding_dim = 32

decoder_hidden_dim = 128
decoder_bottleneck_dim = 64

num_outputs = 10
dropout = 0.10
```

允许构造函数修改，但不要引入 config 系统。

## 21. 不要实现的内容

不要实现：

- CT preprocessing
- CT ResNet
- BiochemistryEncoder
- Dataset / DataLoader
- normalization
- patient split
- Transformer / attention
- evaluation metrics
- optimizer
- scheduler
- checkpoint
- train.py
- evaluate.py
- WandB / TensorBoard

当前只完成：

```text
Fusion + Time Query + Prediction Head + masked loss helper
```

## 22. 依赖

尽量只使用：

```text
torch
argparse
```

## 23. 验收标准

- [ ] 创建 `fusion_query_modal/`
- [ ] 不创建代码子目录
- [ ] 存在 `fusion_time_query.py`
- [ ] ct embedding 默认 `[B,128]`
- [ ] bio embedding 默认 `[B,128]`
- [ ] demographics `[B,2]`
- [ ] time `[B,1]`
- [ ] demo encoder 输出 `[B,32]`
- [ ] concat fusion 输出 baseline state `[B,128]`
- [ ] time MLP 输出 `[B,32]`
- [ ] decoder 输出 `[B,10]`
- [ ] 输出指标顺序固定
- [ ] 支持 batch 与 single sample
- [ ] 输出无 NaN/Inf
- [ ] masked Huber loss 可用
- [ ] CysC mask=0 时该项不影响 loss
- [ ] 不读取 normalization.json
- [ ] 不读取 Excel
- [ ] 不实现 CT encoder
- [ ] 不实现完整训练框架
- [ ] CLI smoke test 全部通过

## 24. 完成后请汇报

请告诉我：

1. 创建了哪些文件
2. `FusionTimeQueryHead` 的实际结构
3. 总参数量
4. batch smoke test shape
5. single-sample smoke test shape
6. masked Huber loss 是否通过
7. CysC mask=0 correctness test 是否通过
8. 当前项目中 CT encoder 是否已经存在
9. 如果存在，实际输出 embedding 维度是多少
10. 是否与默认 128 维一致
11. 如果不一致，如何通过构造参数适配
12. 下一步如何 import，例如：

```python
from fusion_query_modal.fusion_time_query import FusionTimeQueryHead
```

如果发现现有模块接口与本文假设不一致，优先汇报差异，不要擅自修改其他已经构建好的模块。
