# Huashan multimodal forecasting pipeline

This repository trains one longitudinal biochemistry forecasting model with a
configurable CT representation. The baseline remains `roi_25d + resnet18_2d`;
`roi_3d + conv3d` uses the same Dataset, fusion head, training loop, evaluation,
loss, metrics, and patient split.

## CT preparation stages

All stages update one wide `ct_manifest.csv` with one row per patient. Run only a
small representative sample (`--limit 3` to `--limit 5`) before cohort-scale
preprocessing.

```text
DICOM directory or NIfTI
  -> full_ct             canonical complete-coverage LPS NIfTI, HU not windowed
  -> vertebral masks     L2/L3/L4 TotalSegmentator outputs
  -> l2_l4_roi           geometry-preserving lumbar slab and L3 location
  -> roi_25d             fixed [C,H,W] baseline tensor
  -> roi_3d              fixed [D,H,W] physical crop, Dataset adds channel

full_ct -> full_3d       variable-depth preprocessing boundary, not trainable
```

Example commands:

```bash
python -m ct_preparation.prepare_full_ct \
  --input multidataset/huashan_cleaned.xlsx \
  --output-root /data/huashan_ct --relative-paths --limit 3

python -m ct_preparation.segment_vertebrae \
  --manifest /data/huashan_ct/ct_manifest.csv \
  --output-root /data/huashan_ct --relative-paths --limit 3

python -m ct_preparation.extract_l2_l4_roi \
  --manifest /data/huashan_ct/ct_manifest.csv \
  --output-root /data/huashan_ct --relative-paths --limit 3

python -m ct_preparation.build_roi_25d \
  --manifest /data/huashan_ct/ct_manifest.csv \
  --output-root /data/huashan_ct --relative-paths --limit 3

python -m ct_preparation.build_roi_3d \
  --manifest /data/huashan_ct/ct_manifest.csv \
  --output-root /data/huashan_ct --relative-paths --limit 3
```

`roi_3d` defaults to spacing `(x,y,z)=(1.5,1.5,2.0)` mm and stored shape
`(d,h,w)=(64,160,160)`. It resamples to physical spacing, then crops a fixed
field of view around the L3 physical centroid. Missing extent is padded with
`-1024 HU` before applying the `[-200,300] HU` window. It does not resize a
variable anatomical field of view into a fixed pixel matrix. Each processed
patient receives axial/coronal/sagittal preview images for QA.

Image metadata always uses `(x,y,z)` suffixes. NumPy arrays use `(d,h,w)`;
`roi_25d` uses `(c,h,w)`.

## Wide manifest and migration

Canonical fields are grouped by stage:

- source and schema: `manifest_schema_version`, `patient_id`, `source_ct_path`;
- ingestion: `full_ct_*`, selected-series UID/description and geometry;
- segmentation: `l2_mask_path`, `l3_mask_path`, `l4_mask_path`, status/message;
- lumbar ROI: `l2_l4_roi_*`, `l3_center_index`, `l3_centroid_*_mm`;
- representations: `roi_25d_*`, `roi_3d_*`, and `full_3d_*`.

Readers map legacy fields (`nifti_path`, `conversion_status`, `roi_path`,
`roi_status`, `L3_center_slice`, `npy_25d_path`, `build_25d_status`) in memory.
Existing 2.5D arrays do not need regeneration. New writes use canonical names and
preserve unknown manifest columns.

## Configuration and common entry points

CT contracts are version controlled in `configs/ct/`. Experiment files in
`configs/experiments/` reference one CT contract. Incompatible representation,
encoder, shape, or trainability combinations fail before data loading.

```bash
python preflight.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /data/huashan_ct/ct_manifest.csv \
  --ct-root /data/huashan_ct \
  --ct-config configs/ct/roi_3d.yaml

python train.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /data/huashan_ct/ct_manifest.csv \
  --ct-root /data/huashan_ct \
  --config configs/experiments/ct_roi3d_seed42.yaml

python evaluate.py \
  --input multidataset/huashan_cleaned.xlsx \
  --ct-manifest /data/huashan_ct/ct_manifest.csv \
  --ct-root /data/huashan_ct \
  --checkpoint runs/huashan/best.pt
```

Training saves `resolved_config.json`; checkpoints store representation, encoder,
expected shape, and preprocessing version. Evaluation loads this checkpoint
metadata before constructing its Dataset. An optional `--ct-config` is only an
assertion and must match the checkpoint.

Old checkpoints without CT metadata are interpreted as:

```text
roi_25d + resnet18_2d + [3,224,224] + roi25d-v1
```

The Dataset returns `ct`, `ct_path`, and `ct_representation`. Stored `roi_3d`
`[D,H,W]` becomes sample `[1,D,H,W]` and batch `[B,1,D,H,W]`. The baseline is
sample `[3,H,W]` and batch `[B,3,H,W]`.

## Fair comparison and provenance

Normalization metadata records two distinct hashes:

- `cohort_signature`: longitudinal input hash, split file, and effective train
  patient/observation set;
- `representation_signature`: selected representation/version and only its
  patient/path/status records.

Changes to unrelated columns in the wide manifest therefore do not invalidate
biochemical normalization. A fair 2.5D/3D comparison must use identical split
assignments and the same ordered patient intersection for train, validation, and
test. Repair representation failures or evaluate both models on the common
intersection; never tune on the test set.

## Tests

Tests use synthetic arrays/images, do not download pretrained weights, and do not
need patient data:

```bash
python -m unittest discover -s tests -v
python pipeline_smoke_test.py
```

Before a real cohort run, inspect manifest/numeric metadata, all three preview
planes, crop boundaries and padding, then run preflight and one-batch
forward/backward. Raw CT, generated arrays, weights, and run directories must not
be committed.
