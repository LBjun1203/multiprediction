# agents_dataset.md

> 更新说明（2026-08-31）：本目录已从 `biochemistry_dataset/` 重命名为
> `multidataset/`。当前 Dataset 已按新的项目要求接入 CT preprocessing manifest，
> 统一返回 CT、生化、人口学、查询时间和监督目标。下文中“暂不读取 CT”的描述仅为
> 初始生化 Dataset 阶段的历史约束，已由本次多模态 Dataset 要求取代。

## 任务背景

这是华山项目当前数据管线的下一步。

现阶段已经有一个 Excel 清洗脚本 `clean_huashan_dataset.py`，它会把原始宽表整理成适合纵向预测的长表。现在需要在**尚未实现任何 CT 预处理/读取代码**的前提下，先完成：

1. **patient-level 数据划分**
2. **训练集统计量拟合与保存**
3. **PyTorch Dataset**
4. 一个最小可运行的 smoke test

当前研究任务是：

\[
CT_{T0} + Bio_{T0} + Demographics + FutureTime
\rightarrow Bio_t
\]

目前先把 CT 保留为 `ct_path`，**不要读取 CT，不要实现 DICOM/NIfTI 处理，不要实现图像增强，也不要写模型**。

---

# 1. 项目结构要求

请在当前项目根目录下新建一个：

```text
multidataset/
```

**不要再创建任何子文件夹。**

第一版只创建下面两个 Python 文件：

```text
dataset/
├── split_patients.py
└── huashan_dataset.py
```

除此之外不要创建：

- `utils/`
- `configs/`
- `transforms/`
- `preprocess/`
- `tests/`
- `data/`
- YAML 配置系统
- Hydra
- 复杂 class hierarchy

目标是保持代码极其直接，后续我自己可以阅读和修改。

如果需要生成运行产物，可以直接放在 `dataset/` 下，例如：

```text
multidataset/patient_split.csv
multidataset/normalization.json
```

不要移动或修改现有的 `clean_huashan_dataset.py`。

---

# 2. 输入数据

输入是 `clean_huashan_dataset.py` 生成的 `.xlsx` 文件。

默认使用 sheet：

```text
model_long_0_2y
```

该 sheet 中：

- 每一行 = 某位患者的一个未来随访 observation
- 同一患者可以出现多行
- `patient_id` 是患者级 ID
- `ct_path` 是 T0 CT 路径
- `t_days` 是该随访距离 T0 的真实天数
- 当前任务只使用 `0 < t_days <= 730`

不要假设输入文件名固定，所有脚本都应通过 CLI `--input` 指定 Excel 路径。

---

# 3. 生化指标定义

固定顺序如下，**任何地方都不要改变顺序**：

```python
CORE_BIOMARKERS = [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
]

ALL_BIOMARKERS = CORE_BIOMARKERS + ["CysC"]
```

因此最终所有生化 tensor 的第 0~8 位是 9 个核心指标，第 9 位是 CysC。

基线列：

```text
base_HBA1C
base_TG
base_CHO
base_HDL
base_LDL
base_ALB
base_BUN
base_UA
base_CRE
base_CysC
base_CysC_mask
```

目标列：

```text
target_HBA1C
target_TG
target_CHO
target_HDL
target_LDL
target_ALB
target_BUN
target_UA
target_CRE
target_CysC
target_CysC_mask
```

人口学信息：

```text
age
sex_code
```

其中约定：

```text
sex_code:
女 = 0
男 = 1
```

时间：

```text
t_days
```

CT：

```text
ct_path
```

---

# 4. `split_patients.py`

## 4.1 目标

完成**严格 patient-level** 的 train / val / test 划分。

绝对不能按 Excel 行随机划分，因为同一个患者存在多个随访 observation。

错误方式：

```python
train_test_split(df)
```

正确逻辑：

```python
patient_ids = df["patient_id"].unique()
# 先划 patient_id
# 再根据 patient_id 得到 train/val/test observation
```

---

## 4.2 默认划分

第一版使用最简单、可复现的 hold-out：

```text
train = 70%
val   = 15%
test  = 15%
seed  = 42
```

通过 CLI 参数允许修改：

```text
--train-ratio
--val-ratio
--test-ratio
--seed
```

默认值仍然是：

```text
0.70 / 0.15 / 0.15 / 42
```

三者必须相加为 1。

目前先不要实现复杂 stratified split、cross-validation manager 等。

后续正式实验可能会改成 patient-level 5-fold CV，但**当前 Dataset API 不应依赖具体划分方式**，因此未来替换 split 文件即可。

---

## 4.3 输出

保存：

```text
multidataset/patient_split.csv
```

格式必须是一位患者一行：

```text
patient_id,split
10001857,train
10005514,test
10006462,val
...
```

`patient_id` 必须作为字符串处理，不能因为 Excel 数字类型出现 `.0`。

---

## 4.4 必须检查

脚本运行时打印并验证：

### Patient 数量

```text
All patients:
Train patients:
Val patients:
Test patients:
```

### Observation 数量

因为每个患者随访次数不同，还要打印：

```text
Train observations:
Val observations:
Test observations:
```

### 泄漏检查

必须断言：

```python
train_ids & val_ids == set()
train_ids & test_ids == set()
val_ids & test_ids == set()
```

并断言：

```python
train_ids | val_ids | test_ids == all_patient_ids
```

### 简单分布检查

打印每个 split 的：

- patient 数
- observation 数
- 平均每位患者 observation 数
- `t_days` 中位数
- 0–1 年 observation 数
- 1–2 年 observation 数

只打印即可，不需要画图。

---

## 4.5 推荐 CLI

例如：

```bash
python dataset/split_patients.py \
    --input huashan_cleaned.xlsx \
    --sheet model_long_0_2y \
    --output multidataset/patient_split.csv \
    --seed 42
```

---

# 5. `huashan_dataset.py`

这是本任务的核心文件。

它负责：

1. 读取 `model_long_0_2y`
2. 根据 `patient_split.csv` 选择 train / val / test
3. 只使用训练患者拟合 normalization statistics
4. 保存 / 读取 normalization statistics
5. 构建 PyTorch Dataset
6. 暂时只返回 `ct_path`，**不读取 CT**
7. 提供一个 CLI smoke test

---

# 6. Dataset 每个 item 的定义

`__getitem__()` 返回一个 dict，固定采用下面的 key：

```python
{
    "patient_id": str,
    "ct_path": str,

    "bio_base": torch.FloatTensor,       # shape = [10]
    "base_mask": torch.FloatTensor,      # shape = [10]

    "demographics": torch.FloatTensor,   # shape = [2]
    "time": torch.FloatTensor,           # shape = [1]

    "target": torch.FloatTensor,         # shape = [10]
    "target_mask": torch.FloatTensor,    # shape = [10]
}
```

不要把所有输入拼成一个大 tensor。

后续模型会分别使用：

```text
bio_base
demographics
time
CT
```

因此 Dataset 阶段保持语义清楚。

---

# 7. `bio_base`

固定顺序：

```text
[
 HBA1C,
 TG,
 CHO,
 HDL,
 LDL,
 ALB,
 BUN,
 UA,
 CRE,
 CysC
]
```

shape：

```text
[10]
```

dtype：

```python
torch.float32
```

---

# 8. `base_mask`

shape：

```text
[10]
```

前 9 个核心指标在当前清洗数据中应当完整，因此：

```text
base_mask[0:9] = 1
```

最后：

```text
base_mask[9] = base_CysC_mask
```

约定：

```text
1 = observed
0 = missing
```

不要反过来。

---

# 9. `target`

顺序与 `bio_base` 完全一致：

```text
[
 HBA1C,
 TG,
 CHO,
 HDL,
 LDL,
 ALB,
 BUN,
 UA,
 CRE,
 CysC
]
```

shape：

```text
[10]
```

---

# 10. `target_mask`

shape：

```text
[10]
```

当前：

```text
target_mask[0:9] = 1
target_mask[9] = target_CysC_mask
```

这使未来 loss 可以直接写成类似：

```python
loss_per_element = criterion(pred, target)
loss = (loss_per_element * target_mask).sum() / target_mask.sum()
```

本任务不要实现 model/loss，只需保证 Dataset 输出支持这种用法。

---

# 11. CysC 缺失处理

这是必须严格遵守的部分。

## 11.1 Baseline CysC

不能：

```python
NaN -> 0
```

也不能使用全部数据的均值/中位数填补。

必须：

1. 只从 **train patients 的唯一 baseline** 中取有观测的 `base_CysC`
2. 计算训练集 median
3. 保存该 median
4. 对 train / val / test 中缺失的 `base_CysC` 都使用这个 **train median** 填补
5. 同时保留：

```text
base_mask[9] = 0
```

因此模型可以区分：

```text
真实 CysC 值
```

与：

```text
因为缺失而被填补的 CysC 值
```

---

## 11.2 Target CysC

如果 target CysC 缺失：

```text
target_mask[9] = 0
```

最终返回的 `target[9]` 不能是 NaN，否则 batch/loss 很容易传播 NaN。

在完成 target normalization 后：

```text
missing target_CysC -> 0.0
```

即可。

因为：

```text
target_mask[9] = 0
```

未来 loss 会忽略这一项。

注意：

**这个 0.0 是标准化空间中的占位值，不代表真实 CysC=0。**

---

# 12. Normalization：严禁数据泄漏

现有清洗脚本故意没有标准化。

现在由 `huashan_dataset.py` 完成，但是所有统计量都必须**只使用 train patients**。

建议保存：

```text
multidataset/normalization.json
```

---

# 13. Baseline normalization

### 非常重要

`model_long_0_2y` 是 observation-level 长表，所以同一个人的 baseline 会重复很多次。

因此不能直接：

```python
train_df[base_columns].mean()
```

否则随访次数多的患者会被重复计权。

拟合 baseline statistics 前必须：

```python
train_baseline_df = (
    train_df
    .sort_values(...)
    .drop_duplicates(subset="patient_id")
)
```

然后再计算。

---

## 13.1 9 个核心 baseline 生化

对以下每一列：

```text
base_HBA1C
...
base_CRE
```

在**唯一 train patient baseline** 上计算：

```text
mean
std
```

然后：

```python
x_norm = (x - mean) / std
```

若 std 极小或为 0，则安全设为：

```text
std = 1.0
```

---

## 13.2 Baseline CysC

顺序：

```text
train observed CysC
    ↓
计算 train median
    ↓
用 train median 填补缺失
    ↓
在 train unique patients 上计算 mean/std
    ↓
标准化
```

`median / mean / std` 都保存到 normalization JSON。

---

# 14. Target normalization

target statistics 使用：

```text
train split 中的随访 observation
```

进行拟合。

对于核心 9 项：

```text
target_HBA1C ... target_CRE
```

分别计算 train observation 的：

```text
mean
std
```

对于：

```text
target_CysC
```

只使用：

```text
target_CysC_mask == 1
```

的训练 observation 计算：

```text
mean
std
```

缺失 CysC 不得参与统计量计算。

---

# 15. Demographics normalization

返回：

```python
demographics = [age_normalized, sex_code]
```

shape：

```text
[2]
```

### age

只在：

```text
unique train patients
```

上计算：

```text
mean
std
```

train / val / test 都使用同一组 train statistics。

### sex_code

保持：

```text
0 / 1
```

直接转为 float32，不进行标准化。

如果发现 `sex_code` 或 `age` 缺失，第一版直接抛出清楚的 `ValueError`，不要偷偷填补。

---

# 16. Time

暂时不要对 `t_days` 做 train mean/std normalization。

直接：

```python
time_years = t_days / 365.25
```

Dataset 返回：

```python
"time": torch.tensor([time_years], dtype=torch.float32)
```

因此目前：

```text
0 < time <= 约 2.0
```

这种表达既简单，也方便以后做：

- Time2Vec
- Fourier embedding
- MLP time embedding
- Transformer time query

不要在 Dataset 中提前实现这些时间编码。

---

# 17. CT 当前阶段如何处理

**完全不要读取图像。**

当前 Dataset：

```python
"ct_path": str(row["ct_path"])
```

即可。

不要：

- `pydicom.dcmread`
- SimpleITK
- nibabel
- MONAI transforms
- resize
- resample
- crop
- L3 detection
- 2D/2.5D/3D loading
- U-Net
- ResNet

不要检查 `ct_path` 指向的目录是否一定能在当前机器访问。

原因是这些路径可能来自服务器，Codex 当前运行环境未必拥有挂载点。

只要求：

```text
ct_path 非空
```

即可。

未来 CT pipeline 完成以后，再扩展 Dataset。

---

# 18. normalization.json 格式

格式不强制逐字一致，但要保持**简单、可读、纯 JSON**。

建议类似：

```json
{
  "biomarker_order": [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
    "CysC"
  ],
  "base": {
    "HBA1C": {"mean": 0.0, "std": 1.0},
    "...": {},
    "CysC": {
      "median": 0.0,
      "mean": 0.0,
      "std": 1.0
    }
  },
  "target": {
    "HBA1C": {"mean": 0.0, "std": 1.0},
    "...": {},
    "CysC": {"mean": 0.0, "std": 1.0}
  },
  "age": {
    "mean": 0.0,
    "std": 1.0
  }
}
```

JSON 中保存普通 Python float，不要保存 numpy 类型导致序列化失败。

---

# 19. 推荐 API

代码无需完全逐字照搬，但尽量保持接口简单。

例如：

```python
class HuashanMultimodalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        normalization_stats: dict,
    ):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
```

再提供：

```python
def load_longitudinal_table(
    xlsx_path,
    sheet_name="model_long_0_2y",
):
    ...
```

```python
def load_patient_split(split_csv):
    ...
```

```python
def fit_normalization_stats(
    train_df,
):
    ...
```

```python
def save_normalization_stats(stats, path):
    ...
```

```python
def load_normalization_stats(path):
    ...
```

```python
def build_datasets(
    xlsx_path,
    split_csv,
    stats_path,
    sheet_name="model_long_0_2y",
):
    ...
```

最终：

```python
train_dataset, val_dataset, test_dataset = build_datasets(...)
```

即可。

---

# 20. Patient-balanced sampling

同一患者可能有很多 follow-up observation。

如果未来 DataLoader 对所有 observation 等概率抽样，那么随访次数多的患者会对训练贡献更大。

请在 `huashan_dataset.py` 中额外提供一个**非常简单的 helper**：

```python
def make_patient_balanced_sampler(dataset):
    ...
```

逻辑：

对于患者 \(i\) 有 \(n_i\) 个 observation，则该患者每个 observation 权重：

```text
1 / n_i
```

最后使用：

```python
torch.utils.data.WeightedRandomSampler
```

目标是使不同患者总采样权重接近一致。

不要自动强制使用 sampler，只提供 helper 即可。

以后训练代码可以选择：

```python
sampler = make_patient_balanced_sampler(train_dataset)

DataLoader(
    train_dataset,
    batch_size=...,
    sampler=sampler,
    shuffle=False,
)
```

---

# 21. CLI smoke test

`huashan_dataset.py` 本身提供一个非常简单的 CLI 测试功能，不要额外创建 test 文件。

推荐：

```bash
python multidataset/huashan_dataset.py \
    --input huashan_cleaned.xlsx \
    --ct-manifest /path/to/ct_manifest.csv \
    --sheet model_long_0_2y \
    --split-file multidataset/patient_split.csv \
    --stats-file multidataset/normalization.json \
    --fit-stats \
    --smoke-test
```

如果：

```text
--fit-stats
```

存在，则：

1. 只使用 train split
2. 拟合 normalization statistics
3. 保存 `normalization.json`

然后构建 train/val/test Dataset。

---

# 22. Smoke test 必须输出

类似：

```text
Train dataset: xxx observations / xxx patients
Val dataset:   xxx observations / xxx patients
Test dataset:  xxx observations / xxx patients

bio_base shape:      torch.Size([10])
base_mask shape:     torch.Size([10])
demographics shape:  torch.Size([2])
time shape:          torch.Size([1])
target shape:        torch.Size([10])
target_mask shape:   torch.Size([10])

Example patient_id: ...
Example ct_path: ...
Example t_years: ...
```

并执行断言：

```python
assert bio_base.shape == (10,)
assert base_mask.shape == (10,)
assert demographics.shape == (2,)
assert time.shape == (1,)
assert target.shape == (10,)
assert target_mask.shape == (10,)
```

---

# 23. NaN / Inf 检查

Dataset 最终输出的所有数值 tensor：

```text
bio_base
base_mask
demographics
time
target
target_mask
```

必须：

```python
torch.isfinite(tensor).all()
```

即不允许出现：

```text
NaN
+Inf
-Inf
```

如果核心 9 项发现异常缺失，不要偷偷填 0，应该抛出清楚错误并指出：

```text
patient_id
column name
```

只有 CysC 允许按本文件定义的规则进行缺失处理。

---

# 24. Mask 检查

必须验证：

```text
mask 只能包含 0 和 1
```

同时检查：

```text
base_mask[:9] == 1
target_mask[:9] == 1
```

第 10 位：

```text
base_mask[9] == base_CysC_mask
target_mask[9] == target_CysC_mask
```

---

# 25. 数据泄漏检查

这是本任务最重要的验收条件之一。

必须确保：

### Split 层面

同一个 `patient_id` 不能跨：

```text
train / val / test
```

### Normalization 层面

以下统计量只能由 train patient 得到：

```text
baseline biochemical mean/std
baseline CysC median
baseline CysC mean/std
age mean/std
target biochemical mean/std
target CysC mean/std
```

val/test 只允许读取并应用 train statistics。

禁止：

```python
full_df.mean()
full_df.std()
```

然后再切 split。

---

# 26. Observation-level 与 Patient-level 的区别

请在代码注释中明确：

```text
一个 Dataset item = 一个随访 observation
```

因此：

```python
len(dataset)
```

返回 observation 数，不是 patient 数。

但：

```text
train/val/test 划分单位 = patient
```

这是整个数据结构最关键的概念。

---

# 27. 不要在本任务中做的事情

以下全部超出当前任务范围，不要实现：

- CT 读取
- DICOM 处理
- NIfTI 转换
- CT resize/resample
- 腰椎定位
- L3/L2-L4 ROI
- 2D / 2.5D / 3D 图像 Dataset
- 数据增强
- CNN / ResNet / U-Net
- 生化 MLP
- Fusion
- Time2Vec
- Transformer
- loss
- optimizer
- training loop
- validation loop
- WandB
- TensorBoard
- YAML/Hydra
- distributed training
- mixed precision
- 复杂缓存系统

当前任务只把：

```text
cleaned Excel
    ↓
patient split
    ↓
train-only normalization
    ↓
PyTorch Dataset
```

做正确。

---

# 28. 依赖

尽量只使用：

```text
pandas
numpy
openpyxl
scikit-learn
torch
```

不要引入额外第三方库。

如果某个功能纯 Python / pandas 就能实现，不要为了少量代码增加依赖。

---

# 29. 代码风格

要求：

- Python 3.10+
- 使用 `pathlib.Path`
- 函数命名清楚
- 关键逻辑写中文或英文短注释
- 不要过度封装
- 不要写“企业级”框架
- 不要为了设计模式增加 class
- 错误信息要可读
- 所有随机划分必须可复现
- 常量只在 `huashan_dataset.py` 或对应脚本顶部定义即可
- 不要把几百行代码拆成很多文件

---

# 30. 完成后的运行顺序

Codex 完成代码后，请实际执行一次完整 smoke test。

预期流程：

## Step 1：先运行现有清洗脚本

例如：

```bash
python clean_huashan_dataset.py \
    --input try.xlsx \
    --output huashan_cleaned.xlsx
```

如果项目中已经存在清洗后的 Excel，则直接使用，不需要重复生成。

---

## Step 2：patient-level split

```bash
python dataset/split_patients.py \
    --input huashan_cleaned.xlsx \
    --sheet model_long_0_2y \
    --output multidataset/patient_split.csv \
    --seed 42
```

---

## Step 3：拟合 train statistics + Dataset smoke test

```bash
python multidataset/huashan_dataset.py \
    --input huashan_cleaned.xlsx \
    --ct-manifest /path/to/ct_manifest.csv \
    --sheet model_long_0_2y \
    --split-file multidataset/patient_split.csv \
    --stats-file multidataset/normalization.json \
    --fit-stats \
    --smoke-test
```

---

# 31. 最终验收标准

完成后必须满足以下所有条件：

- [ ] 当前项目根目录存在 `dataset/`
- [ ] `dataset/` 内没有任何子目录
- [ ] 存在 `dataset/split_patients.py`
- [ ] 存在 `multidataset/huashan_dataset.py`
- [ ] 能从 `model_long_0_2y` 读取数据
- [ ] patient_id 按字符串处理
- [ ] train/val/test 按 patient 划分
- [ ] 三个 split 无任何 patient overlap
- [ ] 默认 split 为 70/15/15，seed=42
- [ ] normalization 只用 train 数据拟合
- [ ] baseline statistics 对 unique train patients 拟合，避免重复 baseline 加权
- [ ] baseline CysC 用 train median 填补
- [ ] CysC mask：1=observed，0=missing
- [ ] target CysC 缺失时 tensor 中使用 0.0 占位并由 mask 屏蔽
- [ ] 生化输入 shape = `[10]`
- [ ] target shape = `[10]`
- [ ] base mask shape = `[10]`
- [ ] target mask shape = `[10]`
- [ ] demographics shape = `[2]`
- [ ] time shape = `[1]`
- [ ] `time = t_days / 365.25`
- [ ] 所有数值 tensor 无 NaN / Inf
- [ ] Dataset 暂时不读取 CT
- [ ] `ct_path` 作为字符串返回
- [ ] 提供 patient-balanced sampler helper
- [ ] 能保存 / 加载 `normalization.json`
- [ ] CLI smoke test 可以正常运行
- [ ] 不实现任何模型或训练代码
- [ ] 不创建复杂项目结构

---

# 32. 完成后请向我汇报

Codex 完成后，不要只说“done”。

请简要汇报：

1. 实际创建了哪些文件
2. patient split 的患者数与 observation 数
3. normalization.json 是否成功生成
4. Dataset 一个 item 的各 tensor shape
5. CysC 缺失有多少条、mask 是否正常
6. smoke test 是否全部通过
7. 是否发现输入 Excel 的异常字段或异常值
8. 给出我下一步可以直接运行的命令

如果发现数据结构与本文档假设不一致，**优先停止并说明具体列名/问题，不要擅自猜测字段含义。**
