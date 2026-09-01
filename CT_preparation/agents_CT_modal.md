# agents_CT_modal.md

## 任务目标

请为华山项目构建一个**尽量简单、可读、可逐步调试**的 CT 预处理模块。

当前整体研究 Pipeline 为：

```text
T0 CT
  ↓
预训练椎体分割
  ↓
定位 L2 / L3 / L4
  ↓
得到 L2–L4 附近 CT ROI
  ↓
构建 2.5D 输入
  ↓
后续使用 2.5D ResNet / 2.5D U-Net 编码
  ↓
CT embedding
```

本任务只完成：

```text
DICOM / NIfTI
    ↓
标准 NIfTI
    ↓
TotalSegmentator 预训练椎体分割
    ↓
L2/L3/L4 定位
    ↓
L2–L4 ROI
    ↓
2.5D 输入文件
```

**不要实现 ResNet、U-Net、Fusion、训练循环或生化预测。**

---

# 0. 关于解剖区域

本文件按照目前已经确认的方案：

```text
L2–L4 腰椎区域
```

设计。

目标标签：

```text
vertebrae_L2
vertebrae_L3
vertebrae_L4
```

如果未来确认研究对象实际上需要胸椎：

```text
T2–T4
```

只需要修改目标标签列表，整体代码结构不应改变。

---

# 1. 为什么使用预训练分割模型

本项目真正研究目标不是“训练椎体分割模型”，而是：

```text
CT → CT feature → 生化轨迹预测
```

因此椎体分割只是 preprocessing / anatomical localization。

第一版直接使用预训练：

```text
TotalSegmentator
```

定位 L2/L3/L4。

不要：

- 从头训练椎体 U-Net
- 自己制作大规模 vertebra segmentation 标注
- 将 TotalSegmentator 加入 end-to-end training
- 为椎体分割设计复杂训练框架

TotalSegmentator 在本项目中只作为固定的 preprocessing tool。

---

# 2. 项目目录要求

在当前项目根目录下创建：

```text
CT_modal/
```

所有 Python 脚本直接放在该目录中。

**不要在 CT_modal 中再创建代码子文件夹。**

第一版只创建：

```text
CT_modal/
├── convert_dicom.py
├── segment_vertebrae.py
├── extract_l2_l4_roi.py
└── build_25d.py
```

不要创建：

```text
CT_modal/utils/
CT_modal/configs/
CT_modal/models/
CT_modal/transforms/
CT_modal/preprocess/
CT_modal/tests/
```

不要使用：

- Hydra
- YAML 配置系统
- 复杂 class hierarchy
- plugin system
- registry
- factory pattern

代码应该让我以后可以直接打开一个 `.py` 文件阅读和修改。

---

# 3. 生成的数据不要放在代码目录里

脚本通过 CLI 接受：

```text
--output-root
```

例如：

```bash
--output-root /mnt/data/huashan_ct_processed
```

生成的数据可以是：

```text
huashan_ct_processed/
├── 10001857/
│   ├── ct.nii.gz
│   ├── segmentation/
│   │   ├── vertebrae_L2.nii.gz
│   │   ├── vertebrae_L3.nii.gz
│   │   └── vertebrae_L4.nii.gz
│   ├── l2_l4_roi.nii.gz
│   └── l3_25d.npy
├── 10005514/
│   └── ...
└── ct_manifest.csv
```

这里允许生成数据子目录。

限制的是：

> **代码文件全部直接放在 `CT_modal/` 中，不要构建复杂源码目录。**

---

# 4. 输入数据

目前生化 Dataset 已经由：

```text
clean_huashan_dataset.py
```

整理。

CT preprocessing 第一版建议读取清洗后 Excel 的：

```text
baseline
```

sheet。

原因：

```text
baseline = 一位患者一行
```

不会像：

```text
model_long_0_2y
```

那样因为多个 follow-up observation 重复处理同一套 CT。

需要的列至少包括：

```text
patient_id
ct_path
```

如果用户指定其他 sheet，也允许：

```text
--sheet
```

如果 sheet 中同一个 `patient_id` 出现多次：

```python
drop_duplicates(subset="patient_id")
```

但必须先检查同一个 patient 是否对应多个不同的 `ct_path`。

如果同一个 patient 对应多个不同 CT path：

> 不要擅自选择。

必须抛出清楚错误或在 manifest 中标记为异常。

---

# 5. Patient ID

`patient_id` 必须始终当作字符串。

避免：

```text
10001857.0
```

应该最终保持：

```text
10001857
```

这一规则与 `dataset/` 模块保持一致。

---

# 6. 整体执行顺序

目标工作流：

```text
Step 1
DICOM folder
    ↓
convert_dicom.py
    ↓
ct.nii.gz

Step 2
ct.nii.gz
    ↓
segment_vertebrae.py
    ↓
L2/L3/L4 masks

Step 3
CT + L2/L3/L4 masks
    ↓
extract_l2_l4_roi.py
    ↓
l2_l4_roi.nii.gz
    ↓
记录 L3 中心位置

Step 4
l2_l4_roi.nii.gz
    ↓
build_25d.py
    ↓
l3_25d.npy
```

每一步应该可以独立运行和检查。

不要一开始写成一个无法调试的大脚本。

---

# 7. `convert_dicom.py`

## 7.1 目标

将 Excel 中：

```text
ct_path
```

对应的 CT 序列转换成：

```text
NIfTI
```

统一输出：

```text
<output_root>/<patient_id>/ct.nii.gz
```

---

# 8. DICOM 转 NIfTI

优先使用：

```text
dcm2niix
```

不要自己用 pydicom 手工拼 volume。

原因：

- DICOM series 排序复杂
- ImagePositionPatient
- InstanceNumber
- slice spacing
- orientation
- metadata inconsistencies

这些问题尽量交给成熟工具。

建议通过：

```python
subprocess.run(...)
```

调用系统中的：

```text
dcm2niix
```

---

# 9. 输入也可能已经是 NIfTI

`ct_path` 可能未来直接指向：

```text
.nii
.nii.gz
```

因此：

```text
convert_dicom.py
```

应该支持：

### 情况 A

```text
ct_path = DICOM folder
```

执行 dcm2niix。

### 情况 B

```text
ct_path = existing .nii / .nii.gz
```

直接复制/链接到标准输出：

```text
ct.nii.gz
```

或者记录原文件并继续。

不要为了 NIfTI 再进行 DICOM 转换。

---

# 10. dcm2niix 调用要求

转换后必须确认：

```text
输出 NIfTI 数量
```

第一版假定 Excel 的 `ct_path` 已经指向一个明确 CT series。

因此理想情况：

```text
1 patient path → 1 CT volume
```

如果 dcm2niix 得到多个 NIfTI：

> 不要自动选“最大的那个”。

记录：

```text
multiple_nifti_outputs
```

并让该病例进入 QC。

---

# 11. Conversion QC

每位患者记录：

```text
patient_id
source_ct_path
nifti_path
conversion_status
conversion_message
```

推荐状态：

```text
success
failed
skipped_existing
```

如果转换失败：

- 不要删除患者
- 不要伪造输出
- manifest 中记录失败原因
- 后续脚本自动跳过 failed patient

---

# 12. 基础 NIfTI 检查

成功转换后检查：

```text
shape
spacing
affine / direction
finite voxel values
```

至少记录：

```text
size_x
size_y
size_z
spacing_x
spacing_y
spacing_z
```

不要因为 spacing 与其他患者不同就删除。

---

# 13. `segment_vertebrae.py`

## 13.1 目标

输入：

```text
ct.nii.gz
```

使用预训练：

```text
TotalSegmentator
```

只生成：

```text
vertebrae_L2
vertebrae_L3
vertebrae_L4
```

不要为了本任务默认分割 100+ 个器官。

---

# 14. TotalSegmentator

优先调用当前环境中已有：

```text
totalsegmentator
```

CLI。

通过：

```python
subprocess.run(...)
```

调用即可。

在 Codex 实现时：

1. 先检查本机 TotalSegmentator 版本
2. 查看 `totalsegmentator --help`
3. 根据当前版本正确使用 ROI subset 参数

目标语义必须是：

```text
只输出 vertebrae_L2 / vertebrae_L3 / vertebrae_L4
```

不要因为 CLI 版本差异擅自更改目标。

---

# 15. Segmentation 输出

每位患者：

```text
<patient_id>/
└── segmentation/
    ├── vertebrae_L2.nii.gz
    ├── vertebrae_L3.nii.gz
    └── vertebrae_L4.nii.gz
```

---

# 16. Segmentation QC

每个 mask 检查：

```text
file exists
voxel count > 0
shape 与 CT 一致
affine / geometry 与 CT 对齐
```

记录：

```text
L2_voxels
L3_voxels
L4_voxels
```

如果某一个 mask 完全为空：

```text
segmentation_status = failed
```

不要：

- 用 L1 代替 L2
- 用 L5 代替 L4
- 自动猜测椎体
- 根据 slice index 自己补一个假 mask

这类病例需要人工 QC。

---

# 17. 不要求训练/计算 Dice

本项目第一版不要求：

```text
Dice
IoU
Hausdorff
```

因为没有椎体人工 GT。

第一阶段的 QC 方法是：

```text
TotalSegmentator 自动分割
        ↓
数值检查
        ↓
随机人工可视化抽查
```

Codex 当前只实现数值 QC。

不要增加标注工具。

---

# 18. `extract_l2_l4_roi.py`

这是本模块最重要的脚本。

输入：

```text
ct.nii.gz
vertebrae_L2.nii.gz
vertebrae_L3.nii.gz
vertebrae_L4.nii.gz
```

输出：

```text
l2_l4_roi.nii.gz
```

以及 ROI metadata。

---

# 19. 一个非常重要的原则

不要把：

```text
L2/L3/L4 vertebra mask
```

本身作为最终 CT 输入。

椎体 mask 在本项目中只用于：

```text
anatomical localization
```

最终 ROI 必须保留椎体周围的：

- skeletal muscle
- visceral region
- subcutaneous fat
- surrounding soft tissue

因此绝对不能：

```text
CT * vertebra_mask
```

然后只保留骨头。

---

# 20. 第一版 ROI 策略

为了避免过早设计复杂 body segmentation，第一版采用一个非常稳妥的方案。

## Z 方向

使用：

```text
L2 + L3 + L4
```

mask 的 union。

得到 union 在 superior–inferior 方向的范围。

然后上下增加：

```text
margin_mm
```

默认：

```text
20 mm
```

CLI 可配置：

```text
--z-margin-mm 20
```

即：

```text
L2 superior
      ↑ +20 mm

      L2
      L3
      L4

      ↓ +20 mm
L4 inferior
```

---

# 21. X/Y 方向不要只裁椎体 bbox

第一版：

> **保留 CT 原始完整 x/y FOV。**

即只在 z 方向裁剪到 L2–L4 区域。

原因：

如果用 vertebra bounding box 再向 x/y 扩几十 mm，很容易把：

- 腹部脂肪
- 外侧肌肉
- 皮下脂肪

裁掉。

而这些恰恰可能与代谢/生化状态相关。

所以第一版：

```text
x = full FOV
y = full FOV
z = L2–L4 anatomical crop
```

这是一个故意保持简单的设计。

未来如果发现 FOV 差异过大，再增加：

```text
body mask → body bounding box
```

但本任务不要做。

---

# 22. L3 中心定位

同时从：

```text
vertebrae_L3
```

计算 L3 mask centroid。

推荐保存：

```text
L3 centroid physical coordinate
```

而不是只保存原始 voxel index。

因为后续 ROI 会 resample。

metadata 至少包含：

```text
patient_id
L3_centroid_x_mm
L3_centroid_y_mm
L3_centroid_z_mm
```

---

# 23. ROI resampling

为了后续构建一致的 2.5D slice stack，ROI 需要统一 spacing。

第一版默认：

```text
target spacing:
1.5 mm × 1.5 mm × 2.0 mm
```

允许 CLI：

```text
--target-spacing 1.5 1.5 2.0
```

---

# 24. Resampling interpolation

CT image：

```text
linear interpolation
```

mask 如果需要一起 resample：

```text
nearest-neighbor
```

不要对 segmentation mask 使用 linear interpolation。

---

# 25. Orientation

必须保证后续：

```text
axial slice
```

定义稳定。

可以使用：

- SimpleITK
- nibabel canonical orientation

任选一种清楚可靠的方法。

目标是：

```text
不同患者最终 ROI 的 axis 定义一致
```

不要用“数组第三维一定是 z”这种不检查 affine/orientation 的假设。

---

# 26. 输出 ROI

每位患者：

```text
<patient_id>/l2_l4_roi.nii.gz
```

要求：

- float32 或保留合理 CT datatype
- geometry 正确
- resampled spacing 正确
- 不修改原始 NIfTI

---

# 27. ROI metadata

建议输出一个：

```text
roi_metadata.csv
```

至少：

```text
patient_id
roi_path
roi_shape_x
roi_shape_y
roi_shape_z
roi_spacing_x
roi_spacing_y
roi_spacing_z
L3_center_slice
L3_centroid_x_mm
L3_centroid_y_mm
L3_centroid_z_mm
roi_status
roi_message
```

其中：

```text
L3_center_slice
```

是 L3 centroid 在最终 resampled ROI 中对应的最近 axial slice index。

---

# 28. `build_25d.py`

目标：

从：

```text
l2_l4_roi.nii.gz
```

构建最简单的 2.5D 输入。

第一版默认：

```text
3 adjacent axial slices
```

以：

```text
L3 center slice
```

为中心。

例如：

```text
slice z-1
slice z
slice z+1
```

堆成：

```text
[3, H, W]
```

这样未来若使用 ImageNet pretrained ResNet18：

```text
input channels = 3
```

不需要第一版就修改 ResNet 第一层。

---

# 29. 为什么先用 3 slices

这是初版，不是最终研究结论。

优点：

- 简单
- 内存小
- 适配 pretrained ResNet 3-channel input
- 容易可视化
- 容易 debugging

未来可以比较：

```text
1 slice
3 slices
5 slices
multi-level L2/L3/L4
3D
```

但当前不要实现复杂 multi-view sampling。

---

# 30. 允许配置 num_slices

可以提供：

```text
--num-slices
```

要求必须为：

```text
odd integer
```

例如：

```text
1
3
5
7
```

默认：

```text
3
```

如果 num_slices != 3，只需要正常输出 `[C,H,W]`。

不要在这个脚本里修改任何 ResNet。

---

# 31. Slice 边界

如果中心附近不足所需 slice：

可以：

```text
clamp / replicate nearest valid slice
```

不要让程序 index error。

但如果 L3 center 根本不在 ROI 内：

```text
status = failed
```

不要猜。

---

# 32. In-plane resize

最终默认：

```text
224 × 224
```

允许：

```text
--image-size 224
```

第一版直接将 full-FOV axial slice resize 到固定尺寸。

不要：

- random crop
- augmentation
- rotation
- elastic transform

这些属于训练阶段。

---

# 33. CT intensity

2.5D input 需要固定 intensity policy。

默认采用 soft-tissue window：

```text
HU min = -200
HU max = 300
```

CLI：

```text
--window-min
--window-max
```

允许调整。

处理：

```python
x = clip(x, window_min, window_max)
x = (x - window_min) / (window_max - window_min)
```

最终：

```text
[0, 1]
```

---

# 34. 为什么使用固定 HU window

这样：

- 不需要用全数据统计 mean/std
- 不产生 train/test leakage
- preprocessing 可对所有 patient 一次完成
- 结果可复现

不要在本模块中计算：

```text
全 cohort mean/std
```

---

# 35. 输出 2.5D

默认：

```text
<patient_id>/l3_25d.npy
```

保存：

```text
numpy float32
shape = [3, 224, 224]
```

如果用户设置其他 num_slices/image_size，则 shape 相应变化。

不要保存 torch `.pt` 作为唯一格式。

`.npy` 更通用、更容易检查。

---

# 36. 2.5D metadata

manifest 中记录：

```text
patient_id
input_roi_path
npy_path
num_slices
center_slice
image_size
window_min
window_max
min_value
max_value
build_25d_status
build_25d_message
```

---

# 37. Manifest

整个 CT 模块最终应该维护一个简单：

```text
ct_manifest.csv
```

一位患者一行。

可以由每一步更新相应列。

建议包括：

```text
patient_id
source_ct_path

nifti_path
conversion_status
conversion_message

L2_mask_path
L3_mask_path
L4_mask_path
L2_voxels
L3_voxels
L4_voxels
segmentation_status
segmentation_message

roi_path
L3_center_slice
roi_status
roi_message

npy_25d_path
build_25d_status
build_25d_message
```

不要使用数据库。

CSV 足够。

---

# 38. Resume / skip existing

TotalSegmentator 很耗时。

每个脚本都尽量支持：

```text
--skip-existing
```

如果目标文件已经存在并且基本检查通过：

```text
skip
```

并记录：

```text
skipped_existing
```

不要每次重复跑 TotalSegmentator。

---

# 39. 单患者调试

所有脚本最好支持：

```text
--patient-id
```

这样我可以先单独测试一个患者。

---

# 40. limit 参数

为了 smoke test，建议支持：

```text
--limit 3
```

表示只处理前 3 位 patient。

不要为了测试直接默认跑全 cohort。

---

# 41. 错误处理原则

CT preprocessing 很容易遇到个别坏病例。

所以：

> **单个患者失败不能让整个 cohort 的脚本立即退出。**

每位患者使用：

```python
try:
    ...
except Exception as e:
    ...
```

记录：

```text
failed
error message
```

然后继续下一位患者。

但以下情况应该直接终止整个程序：

- 输入 Excel 不存在
- 必要列不存在
- dcm2niix executable 不存在（convert 阶段）
- TotalSegmentator executable 不存在（segmentation 阶段）
- output root 无法创建
- manifest 格式完全错误

---

# 42. 不允许静默失败

禁止：

```python
except:
    pass
```

所有失败必须：

```text
打印 patient_id
打印错误
写入 manifest
```

---

# 43. CT 路径检查

可以检查：

```python
Path(ct_path).exists()
```

如果当前机器不存在：

```text
conversion_status = failed
conversion_message = source path not found
```

不要自动修改路径字符串。

不要尝试猜测服务器 mount path。

---

# 44. Preprocessing 与 train/test split

这一 CT preprocessing 使用：

- 固定 pretrained TotalSegmentator
- 固定 HU window
- 每个 patient 自身的 deterministic resampling

因此：

> 可以在 patient split 之前处理全部患者。

这不属于 train/test leakage。

但是本任务禁止增加任何：

```text
dataset-wide learned CT statistics
```

例如：

```text
all patient CT mean
all patient CT std
PCA
learned normalization
```

如果未来需要这些，必须只在 train split 上拟合。

---

# 45. 数据完整性

任何时候都不要修改：

```text
原始 DICOM
原始 clean Excel
原始 ct_path
```

所有输出都写到：

```text
output_root
```

---

# 46. 建议依赖

尽量只使用：

```text
pandas
numpy
openpyxl
nibabel
SimpleITK
torch
```

外部程序：

```text
dcm2niix
totalsegmentator
```

不要求同时使用 nibabel 和 SimpleITK。

如果一个库可以完成 orientation / resample / read-write，就尽量统一。

---

# 47. 推荐优先使用 SimpleITK 的场景

如果 Codex 熟悉：

```text
SimpleITK
```

可以主要使用它完成：

- NIfTI read
- spacing
- direction
- physical coordinate
- resampling
- crop
- write

这样可以减少手工处理 affine 的错误。

但不强制。

---

# 48. 不要做的事情

本任务明确不要实现：

- vertebra segmentation training
- nnU-Net training
- U-Net training
- ResNet training
- CT encoder
- feature extraction network
- data augmentation
- random crop
- mixup
- contrastive learning
- self-supervised learning
- CT、生化 fusion
- Time Query
- loss
- optimizer
- DataLoader training loop
- GPU distributed training
- visualization GUI
- 3D Slicer plugin
- web UI
- database

---

# 49. 代码风格

要求：

- Python 3.10+
- `pathlib.Path`
- `argparse`
- 函数命名直观
- 不过度封装
- 少量必要注释
- 关键坐标/orientation逻辑解释清楚
- 日志至少能看出当前 patient
- 代码优先可读性，而非炫技

---

# 50. 推荐 CLI

## convert

```bash
python CT_preparation/convert_dicom.py \
    --input huashan_cleaned.xlsx \
    --sheet baseline \
    --output-root /mnt/data/huashan_ct_processed \
    --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
    --limit 3
```

## segmentation

```bash
python CT_preparation/segment_vertebrae.py \
    --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
    --output-root /mnt/data/huashan_ct_processed \
    --limit 3
```

## ROI

```bash
python CT_preparation/extract_l2_l4_roi.py \
    --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
    --output-root /mnt/data/huashan_ct_processed \
    --target-spacing 1.5 1.5 2.0 \
    --z-margin-mm 20 \
    --limit 3
```

## 2.5D

```bash
python CT_preparation/build_25d.py \
    --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
    --output-root /mnt/data/huashan_ct_processed \
    --num-slices 3 \
    --image-size 224 \
    --window-min -200 \
    --window-max 300 \
    --limit 3
```

---

# 51. 第一轮 smoke test

Codex 完成以后，不要直接运行全 cohort。

选择：

```text
1–3 个 patient
```

完成全流程：

```text
DICOM
 ↓
NIfTI
 ↓
TotalSegmentator
 ↓
L2/L3/L4 masks
 ↓
ROI
 ↓
2.5D npy
```

---

# 52. Smoke test 必须检查

## NIfTI

打印：

```text
patient_id
CT shape
CT spacing
```

## masks

打印：

```text
L2 voxel count
L3 voxel count
L4 voxel count
```

并断言均：

```text
> 0
```

## ROI

打印：

```text
ROI shape
ROI spacing
L3 center slice
```

检查：

```text
0 <= L3_center_slice < ROI axial slice count
```

## 2.5D

加载 `.npy`：

```python
arr = np.load(...)
```

默认断言：

```python
arr.shape == (3, 224, 224)
arr.dtype == np.float32
np.isfinite(arr).all()
arr.min() >= 0
arr.max() <= 1
```

---

# 53. 关于 axial axis

不要直接硬编码：

```python
volume[:, :, z]
```

除非代码已经明确将影像 canonicalize 到固定 orientation，并确认第三维就是 superior-inferior。

这是 CT preprocessing 中很容易出错的一点。

Codex 必须在代码注释中明确：

```text
经过什么 orientation 标准化后，
哪个 axis 被定义为 axial slice axis。
```

---

# 54. 人工 QC 的准备

当前不要求制作 GUI，但建议 `extract_l2_l4_roi.py` 或 `build_25d.py` 可以提供一个：

```text
--save-preview
```

可选参数。

如果实现，不要复杂。

只需保存：

```text
L3 center axial PNG
```

例如：

```text
<patient_id>/l3_preview.png
```

用于肉眼检查：

- 是否确实在 L3 附近
- 是否包含腹部/背部软组织
- 是否发生上下颠倒
- resize 是否异常

如果实现 preview，需要：

```text
windowed image
```

不要使用原始 HU 直接保存导致全黑。

**这个功能是可选，不是必须。**

---

# 55. 失败病例策略

如果：

```text
L2/L3/L4 任一 mask 缺失
```

该 patient：

```text
segmentation_status = failed
```

后续 ROI / 2.5D：

```text
skip
```

不要自动删除 Excel 中的患者。

最终通过 manifest 决定哪些 patient 可用于模型。

---

# 56. 未来与 `dataset/` 模块连接方式

以后 PyTorch Dataset 中：

当前：

```python
"ct_path": original_ct_path
```

将逐步替换为：

```python
"ct_25d_path": npy_25d_path
```

或 Dataset 根据：

```text
patient_id
```

从：

```text
ct_manifest.csv
```

找到：

```text
npy_25d_path
```

但：

> **当前 CT_modal 任务不要修改已有 `multidataset/huashan_dataset.py`。**

先把 CT preprocessing 独立跑通。

---

# 57. 第一阶段最终可用 patient

未来一个 patient 是否能进入 multimodal model，至少要求：

```text
生化 Dataset 中存在
AND
CT conversion success
AND
L2/L3/L4 segmentation success
AND
ROI success
AND
2.5D build success
```

但当前不要写最终 multimodal merge script。

后续单独完成。

---

# 58. 非常重要：不要擅自删患者

CT 失败患者只在：

```text
ct_manifest.csv
```

中标记。

不要：

- 从清洗 Excel 删除
- 修改 patient_split.csv
- 自动重新划 train/val/test

后续我们会根据 CT 成功率决定如何处理 split。

---

# 59. 关于 TotalSegmentator 失败率

第一版先统计。

完成全 cohort 后汇报：

```text
Total patients
conversion success
segmentation success
L2 missing
L3 missing
L4 missing
ROI success
2.5D success
```

如果失败率较低：

```text
继续使用 TotalSegmentator
```

如果明显有大量病例失败，再决定是否：

```text
人工修正
或
fine-tune nnU-Net
```

不要在当前任务提前实现 fine-tuning。

---

# 60. 完整验收标准

完成后应满足：

- [ ] 创建 `CT_modal/`
- [ ] `CT_modal/` 内没有代码子目录
- [ ] 存在 `convert_dicom.py`
- [ ] 存在 `segment_vertebrae.py`
- [ ] 存在 `extract_l2_l4_roi.py`
- [ ] 存在 `build_25d.py`
- [ ] 可以读取清洗后 Excel 的 `baseline`
- [ ] patient_id 始终是字符串
- [ ] 每位 patient 只处理一次 CT
- [ ] DICOM 使用 dcm2niix 转换
- [ ] 已有 NIfTI 可以直接使用
- [ ] 多 NIfTI 输出不会擅自选择
- [ ] TotalSegmentator 只定位 L2/L3/L4
- [ ] 三个 mask 都检查 voxel count
- [ ] mask 与 CT geometry 对齐
- [ ] 椎体 mask 只作为定位，不将 CT 限制为骨组织
- [ ] ROI 默认保留 full x/y FOV
- [ ] ROI z 范围基于 L2–L4 union + margin
- [ ] 保存 L3 physical centroid
- [ ] ROI resample 到统一 spacing
- [ ] CT resample 使用 linear
- [ ] mask resample 使用 nearest neighbor
- [ ] orientation 明确且 axial axis 稳定
- [ ] 默认 target spacing = 1.5 × 1.5 × 2.0 mm
- [ ] 默认 z margin = 20 mm
- [ ] 默认 2.5D = L3 中心连续 3 张 axial slices
- [ ] 默认 resize = 224×224
- [ ] 默认 HU window = [-200, 300]
- [ ] 2.5D 输出 float32 `[3,224,224]`
- [ ] 2.5D 数值范围 `[0,1]`
- [ ] 全流程有 manifest
- [ ] 单病例失败不会停止全 cohort
- [ ] 所有错误都写入 manifest
- [ ] 支持 `--patient-id`
- [ ] 支持少量病例 smoke test
- [ ] TotalSegmentator 已有结果可以 skip
- [ ] 不修改原始 DICOM
- [ ] 不修改原始 Excel
- [ ] 不修改已有 `dataset/` 模块
- [ ] 不实现 ResNet/U-Net 模型
- [ ] 不实现训练代码

---

# 61. Codex 完成后必须汇报

不要只回复“完成”。

请告诉我：

1. 创建了哪些 `.py`
2. 实际使用的 TotalSegmentator CLI 命令
3. 实际使用的 dcm2niix 命令
4. smoke test 使用了哪几个 patient
5. 每个病例原始 CT shape / spacing
6. L2/L3/L4 mask voxel count
7. ROI shape / spacing
8. L3 center slice 是否正确落在 ROI 中
9. `.npy` shape / dtype / min / max
10. 是否有失败病例
11. 失败原因
12. manifest 保存在哪里
13. 下一步我应该运行的完整命令

如果当前机器：

- 没有 dcm2niix
- 没有 TotalSegmentator
- 没有 GPU
- 无法访问 `ct_path`

请明确说明。

**不要伪造 smoke test 结果。**
