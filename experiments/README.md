# Add-only baseline and ablation experiments

本目录为现有 Huashan pipeline 增加一个旁路实验层；原有数据集、CT 模型和
`train.py` 均不修改。

## 公平比较契约

- 所有方法先调用现有 `build_datasets()`，使用同一 CT 可用患者队列、同一患者级
  train/val/test 划分。
- `runs/comparisons/v1/shared/normalization.json` 只在训练患者上拟合一次；此后
  Tabular 与 Full 都复用该文件。
- Tabular 输入固定为 23 维：10 个标准化基线生化值、10 个可用性 mask、标准化
  年龄、性别编码和查询时间（年）。它不读取 CT 数组。
- Full 是原项目模型，仍通过原有 `train.py` 训练。
- 所有最终指标只在固定 test split 计算；Ridge 的 alpha 只由 validation masked
  Huber loss 选择。
- 训练集重复随访通过 `1 / 该患者训练观测数` 权重处理：神经网络使用原有患者均衡
  sampler，Ridge 使用同义的加权最小二乘。

## 第一次运行：seed 42

以下命令均从项目根目录运行。第一步生成共享归一化文件，并一次性计算 Mean、
Persistence 和 Ridge：

```powershell
python experiments/evaluate_classical_baselines.py `
  --input multidataset/huashan_cleaned.xlsx `
  --ct-manifest test/ct_manifest.csv `
  --split-file multidataset/patient_split.csv `
  --stats-file runs/comparisons/v1/shared/normalization.json `
  --output-dir runs/comparisons/v1 `
  --fit-stats
```

训练 CT-free Tabular 模型：

```powershell
python experiments/train_tabular.py `
  --input multidataset/huashan_cleaned.xlsx `
  --ct-manifest test/ct_manifest.csv `
  --split-file multidataset/patient_split.csv `
  --stats-file runs/comparisons/v1/shared/normalization.json `
  --output-dir runs/comparisons/v1/tabular/seed42 `
  --seed 42 --epochs 50 --patience 10
```

用现有 pipeline 训练 Full 模型。这里必须传 `--reuse-stats`，否则原 `train.py`
会重新拟合并覆盖统计文件：

```powershell
python train.py `
  --input multidataset/huashan_cleaned.xlsx `
  --ct-manifest test/ct_manifest.csv `
  --split-file multidataset/patient_split.csv `
  --stats-file runs/comparisons/v1/shared/normalization.json `
  --output-dir runs/comparisons/v1/full/seed42 `
  --reuse-stats --seed 42 --epochs 50 --patience 10
```

最后统一汇总：

```powershell
python experiments/compare_results.py --root runs/comparisons/v1
```

输出包括：

- `summary/comparison_overall.csv`：每个方法的总体指标；
- `summary/comparison_per_biomarker.csv`：每个生化指标的误差、偏倚、相关系数和 R²；
- `summary/comparison_metrics.png`：标准化 RMSE 与 R² 对比图；
- `summary/cohort_audit.json`：各结果是否使用相同且顺序一致的测试行。

经典基线和 Tabular 脚本均默认拒绝覆盖已有结果；经典基线只有在显式传入
`--overwrite` 时才会替换其结果文件。Full 沿用原训练入口，没有这个保护，因此重复
实验请改用新的输出目录，如 `seed43`、`seed44`。

## 各基线的定义

- `mean`：每个指标恒预测训练集目标均值；在标准化空间中就是 0。
- `persistence_imputed`：预测值等于该患者基线值。CysC 基线缺失时使用训练集
  中位数，测试目标 mask 不变。
- `persistence_observed`：预测仍等于基线值，但 CysC 只在“基线和目标都观测到”时
  计分；它用于区分插补带来的影响。由于有效元素数可能不同，其总体分数不能和其他
  方法不加说明地直接排序。
- `ridge`：对每个目标分别拟合带截距的加权岭回归；输入就是上述 23 维 Tabular
  特征，截距不惩罚，alpha 由验证集选择。
- `tabular`：非线性 CT-free 神经网络，保留 Full 中的生化编码、人口学编码、时间
  编码和预测头思想。
- `full`：现有 CT + 生化 + 人口学 + 时间模型。

## 多随机种子

seed 42 验证成功后，再分别以 43 和 44 训练 Tabular、Full，输出到对应的
`seed43`、`seed44` 目录。经典基线是确定性的，不需要重复。汇总脚本会自动递归发现
所有完整结果。

## 快速自检

```powershell
python experiments/smoke_test.py
```

该测试验证 mask 不参与损失、Ridge 闭式解和 Tabular 前向/反向传播。它不读取真实
数据，也不替代真实队列上的 seed 42 验证。
