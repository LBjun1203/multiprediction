# Huashan multimodal forecasting pipeline

This project predicts ten future biochemical markers from an L3-centered CT,
baseline biochemistry, demographics, and a future query time. Train/validation/test
assignment is patient-level, while each Dataset item is one follow-up observation.

## Fixed model contract

- Biomarker order: `HBA1C, TG, CHO, HDL, LDL, ALB, BUN, UA, CRE, CysC`.
- CT model input: `float32 [B,3,H,W]`, values in `[0,1]`, `H,W >= 32`.
- Default CT input: three L3-centered axial slices at `224 x 224`.
- Biochemistry and masks: `[B,10]`; `1=observed`, `0=missing`.
- Demographics: `[normalized_age, sex_code]`, where female=0 and male=1.
- Time: `[B,1]` in years (`t_days / 365.25`).
- Prediction: ten standardized future biomarkers.

The current end-to-end model intentionally supports exactly three CT slices. The CT
preparation script may generate other odd slice counts for future encoders, but those
arrays are rejected by the present training pipeline.

## Environment

Create an isolated Python environment and install a PyTorch build compatible with the
training machine. Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

CT preparation additionally requires:

- `dcm2niix` on `PATH` for DICOM conversion;
- the `totalsegmentator` or `TotalSegmentator` CLI on `PATH`.

The exact PyTorch, CUDA, TotalSegmentator, and driver versions used for a formal
experiment should be captured with the resulting checkpoint/run artifacts.

## 1. CT preparation

Start with a small `--limit` and inspect the manifest before processing the cohort.

```bash
python CT_preparation/convert_dicom.py \
  --input multidataset/huashan_cleaned.xlsx \
  --sheet baseline \
  --output-root /mnt/data/huashan_ct_processed \
  --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --relative-paths \
  --limit 3

python CT_preparation/segment_vertebrae.py \
  --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --output-root /mnt/data/huashan_ct_processed \
  --relative-paths \
  --limit 3

python CT_preparation/extract_l2_l4_roi.py \
  --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --output-root /mnt/data/huashan_ct_processed \
  --target-spacing 1.5 1.5 2.0 \
  --z-margin-mm 20 \
  --relative-paths \
  --limit 3

python CT_preparation/build_25d.py \
  --manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --output-root /mnt/data/huashan_ct_processed \
  --num-slices 3 \
  --image-size 224 \
  --window-min -200 \
  --window-max 300 \
  --save-preview \
  --relative-paths \
  --limit 3
```

Remove `--limit` only after the small run passes numerical and visual QC. Failures
remain in `ct_manifest.csv`; patients are never silently deleted or reassigned.

## 2. Dataset and normalization

Fit normalization on the final CT-eligible train cohort and run the Dataset checks:

```bash
python multidataset/huashan_dataset.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --split-file multidataset/patient_split.csv \
  --stats-file multidataset/normalization.json \
  --fit-stats \
  --smoke-test
```

New normalization files contain hashes of the split, CT manifest, and effective train
patient cohort. Legacy JSON files are accepted with a warning but should not be used
for a formal experiment.

For a manifest containing relative `npy_25d_path` values, pass
`--ct-root /mnt/data/huashan_ct_processed`. Relative paths otherwise resolve against
the manifest directory.

## 3. Read-only preflight

```bash
python preflight.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --split-file multidataset/patient_split.csv \
  --stats-file multidataset/normalization.json \
  --report runs/preflight.json
```

Preflight checks patient leakage, CT eligibility by split, every CT array's shape,
dtype/range, CysC coverage, normalization provenance, and upstream
`comparison_status`. It reports external CT tools but does not require them for model
training.

## 4. Integration smoke test

The integration test creates temporary synthetic arrays and performs a real forward,
backward, optimizer step, and checkpoint reload without downloading pretrained
weights:

```bash
python pipeline_smoke_test.py
```

## 5. Training

The default baseline loads ImageNet ResNet18 weights, freezes the CT backbone, uses
patient-balanced sampling, and fits normalization on the final CT-eligible train
cohort:

```bash
python train.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --split-file multidataset/patient_split.csv \
  --stats-file multidataset/normalization.json \
  --output-dir runs/baseline \
  --batch-size 16 \
  --epochs 50 \
  --learning-rate 1e-3 \
  --patience 10 \
  --device auto
```

Use `--finetune-ct-backbone` only for a deliberate fine-tuning experiment. Use
`--no-pretrained` for offline/debug runs, `--amp` for CUDA mixed precision, and
`--resume runs/baseline/last.pt` to continue a compatible run. `--reuse-stats` is
available only when the stored cohort metadata matches the current inputs.

Training writes `best.pt`, `last.pt`, `history.json`, `test_metrics.json`, and
`test_predictions.csv` into the output directory.

## 6. Evaluation

```bash
python evaluate.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /mnt/data/huashan_ct_processed/ct_manifest.csv \
  --checkpoint runs/baseline/best.pt \
  --split test \
  --predictions runs/baseline/test_predictions_reproduced.csv \
  --metrics runs/baseline/test_metrics_reproduced.json
```

Metrics include masked Huber loss and per-biomarker standardized/clinical-unit MAE
and RMSE. CysC is evaluated only where its target mask equals one.

## Scope

The current version deliberately excludes cross-validation, attention models, CT
augmentation, segmentation fine-tuning, distributed training, experiment tracking,
and target-free prospective clinical inference. Those should be added only after the
baseline pipeline is validated on real CT data.
