# Huashan CT-biochemistry contrastive pretraining refactor

## Mission

Extend the Huashan repository with a recoverable, configuration-driven pipeline
for contrastive pretraining from roughly 7,000 paired same-timepoint L2-L4 CT
volumes and biochemical measurements. Transfer the pretrained CT encoder into
the existing longitudinal model, whose downstream task predicts future
biochemistry from baseline CT, baseline biochemistry, demographics, and query
time.

Preserve the existing 2.5D baseline, lightweight Conv3D baseline, longitudinal
training/evaluation behavior, and legacy checkpoint compatibility. Contrastive
pretraining is an independent task; do not merge its control flow into the
longitudinal training loop.

The user has confirmed these decisions:

1. The first pretraining input is the existing fixed-shape, L3-centered L2-L4
   `roi_3d`, stored `[D,H,W]` and returned `[1,D,H,W]`.
2. Add a 3D ResNet18-class encoder as the main pretraining backbone. Keep the
   current lightweight Conv3D encoder as a selectable baseline.
3. Train CT and biochemistry encoders jointly with a cross-modal contrastive
   objective. The CT encoder is the primary transferable artifact.
4. Projection heads belong to pretraining and are not transferred to the
   downstream forecast model.
5. The roughly 7,000 same-timepoint pairs teach CT-biochemistry associations;
   only the smaller longitudinal cohort teaches temporal change and future
   prediction.

## Scientific scope

The downstream task is conceptually:

```text
(baseline CT, baseline biochemistry, demographics, query time)
    -> future biochemistry
```

The pretraining task is:

```text
same-patient CT at T0 <-> biochemistry at T0
```

Do not claim that same-timepoint contrastive pretraining learns future disease
progression. It initializes the CT encoder with abdominal CT features related to
biochemical phenotype. Longitudinal supervision must determine whether those
features improve future prediction after baseline biochemistry is known.

The eventual scientific comparison should include:

```text
1. baseline biochemistry + demographics + time, without CT
2. randomly initialized 3D ResNet18 CT encoder
3. same-timepoint CT -> biochemistry regression pretraining
4. CT-biochemistry contrastive pretraining
```

Keep architecture, cohort, splits, preprocessing, downstream head, and
fine-tuning schedule matched wherever possible.

## Repository state and version safety

At the time this file was created, the repository was on:

```text
branch: refactor/ct-representation-pipeline
HEAD:   c28708c docs(ct): document representations migration and QA
```

The worktree was not clean. Observed paths included:

```text
modified:
  CT_preparation/manifest.py
  tests/test_ct_pipeline.py
  train.py

untracked:
  AGENTS.md
  clean_huashan_dataset.py
  huashan_cleaned_7000.xlsx
```

Treat all existing changes as user work. Before implementation:

1. Read repository-root `AGENTS.md` completely and follow it together with this
   file. If instructions conflict, stop and report the exact conflict.
2. Inspect `git status`, active branch, recent commits, and current diffs. Do
   not reset, discard, overwrite, stash, or silently include existing changes.
3. Determine which current modifications complete the CT representation
   refactor. Run appropriate tests before proposing a clean recovery point.
4. Do not commit `huashan_cleaned_7000.xlsx`, raw CT, arrays, checkpoints, run
   outputs, or identifiable patient data. Confirm these paths are ignored
   without broadly ignoring source/configuration files.
5. Do not begin from an ambiguous dirty state. Preserve the current code in a
   reviewable commit only after inspection. If ownership or inclusion of a
   dirty file is unclear, report it rather than guessing.
6. After the CT representation pipeline is clean and verified, create:

   ```text
   feat/ct-bio-contrastive-pretraining
   ```

   A separate Git worktree is preferred if it avoids mixing changes with the
   current checkout. Do not create competing implementation branches.
7. Recommend an annotated recovery tag such as
   `ct-representation-pipeline-v1` on the verified pre-contrastive commit. Do
   not push, tag, or change remote state unless the user explicitly requests it.

Keep these recovery points usable:

```text
main at 73dba1b                     -> original 2.5D baseline
refactor/ct-representation-pipeline -> configurable 2.5D/3D pipeline
feat/ct-bio-contrastive-pretraining -> new pretraining work
```

## Existing architecture to preserve

Inspect these files before editing:

```text
ct_modal/encoder_2d.py
ct_modal/encoder_3d.py
ct_modal/factory.py
ct_modal/config.py
ct_preparation/manifest.py
multidataset/huashan_dataset.py
biochemistry_modal/biochemistry_encoder.py
multimodal_model.py
train.py
evaluate.py
```

Preserve these contracts:

- `roi_25d` sample `[3,H,W]` with the 2D ResNet18 path.
- `roi_3d` stored `[D,H,W]`, sample `[1,D,H,W]`, batch `[B,1,D,H,W]`.
- CT encoders return `[B,ct_embedding_dim]`.
- Dataset-to-model key remains `batch["ct"]`.
- The future model continues to own CT encoder, biochemistry encoder,
  demographic/time features, fusion, and output head.
- `--resume` continues to mean full longitudinal-training resumption, including
  optimizer and epoch state.
- Existing checkpoint loading and legacy `roi_25d` defaults remain valid.

Do not duplicate `train.py`, `evaluate.py`, or the full forecast pipeline into
3D- or pretraining-specific copies.

## Target source structure

Add an independent pretraining package:

```text
pretraining/
  __init__.py
  paired_dataset.py
  contrastive_model.py
  losses.py
  checkpoint.py

configs/pretraining/
  ct_bio_roi3d_resnet18.yaml

pretrain_ct_bio.py
```

Add/refactor the 3D ResNet under `ct_modal/`, for example:

```text
ct_modal/
  encoder_3d.py       # existing lightweight Conv3D remains
  resnet3d.py         # 3D ResNet18
  factory.py
```

Keep modules small and task-specific. Share stable contracts and utilities; do
not build a generic training framework prematurely.

## 3D ResNet18 requirements

Implement a native PyTorch 3D ResNet18-style encoder using `BasicBlock3D` with
stage depths `[2,2,2,2]`. It must:

- accept single-channel `[B,1,D,H,W]` input;
- support the current `roi_3d` contract, presently expected to be
  `[B,1,64,160,160]` for `roi3d-l2l4-v1`;
- return `[B,embedding_dim]`;
- expose `backbone_frozen`, `set_backbone_trainable()`, and frozen
  normalization behavior consistent with existing encoders;
- validate rank, channels, floating dtype, finiteness, range, and device;
- use adaptive global pooling rather than hard-coded terminal spatial size;
- allow configurable base channel width when needed for memory, and save the
  resolved width;
- initialize deterministically under the repository seed policy;
- require no network download during tests.

Keep the current lightweight encoder registered as `conv3d`. Register the new
encoder explicitly as `resnet18_3d`; do not change what `conv3d` means.

Do not describe the 3D ResNet as pretrained until compatible local weights are
loaded. If the current `pretrained` boolean becomes ambiguous between ImageNet
2D weights and local 3D initialization, refine the configuration rather than
overloading that boolean.

## Same-timepoint paired Dataset

Create a Dataset specifically for the roughly 7,000 same-timepoint pairs. Do
not add `mode="pretrain"` conditionals throughout the longitudinal Dataset.

Each sample should contain at least:

```python
{
    "ct": FloatTensor[1,D,H,W],
    "bio": FloatTensor[num_biomarkers],
    "bio_mask": FloatTensor[num_biomarkers],
    "patient_id": str,
    "ct_path": str,
}
```

Reuse stable definitions for:

- `CTInputSpec` and representation lookup;
- manifest path resolution and successful status handling;
- CT shape, dtype, finiteness, and value-range validation;
- canonical biomarker order;
- train-only biochemical normalization conventions.

The paired Dataset must not require future targets, query time, or longitudinal
rows. Explicitly validate:

- one resolved CT representation per training sample;
- canonical patient identifiers;
- consistent biochemical units and columns;
- missing values represented by `bio_mask`, not confused with observed zero;
- no non-finite model inputs after normalization/filling;
- exact CT preprocessing version and shape;
- duplicate and repeated-patient behavior.

If several rows or visits exist for one patient, do not treat them as independent
negatives by accident. Document a deterministic policy: select one intended
pair per patient, or use patient-aware multi-positive/sampling logic. The first
implementation may require one pair per patient if verified in the cleaned data.

## Leakage and split policy

This is mandatory because the downstream cohort is small.

1. All splits are patient-level.
2. A downstream validation/test patient must not occur in pretraining train
   data, even at another date.
3. For a fixed downstream split, use the 7,000-pair pool after excluding all
   downstream validation/test patients.
4. For downstream K-fold cross-validation, a strict estimate requires
   fold-specific pretraining exclusion or a locked external pretraining cohort.
   If one encoder has seen later validation/test patients, label that analysis
   transductive; do not present it as leakage-free.
5. Fit biochemical normalization on pretraining train patients only and save
   its provenance.
6. Do not use downstream test performance to select pretraining epochs,
   temperature, backbone width, or fine-tuning policy.

Provide preflight output with pair count, unique patients, missingness by
biomarker, CT eligibility failures, duplicate-patient count, and overlap with
downstream train/validation/test patient lists.

## Contrastive model and loss

Implement a model conceptually equivalent to:

```python
class CTBioContrastiveModel(nn.Module):
    ct_encoder
    bio_encoder
    ct_projector
    bio_projector
    logit_scale_or_temperature
```

Forward path:

```text
CT -> CT encoder -> CT feature -> CT projector -> normalized z_ct
bio + mask -> bio encoder -> bio feature -> bio projector -> normalized z_bio
```

Projection heads are independent modules. Save their shape/architecture in the
resolved configuration. Use symmetric CT-tabular InfoNCE initially:

```text
similarity = z_ct @ z_bio.T / temperature
loss = (cross_entropy(similarity, diagonal_labels)
      + cross_entropy(similarity.T, diagonal_labels)) / 2
```

Requirements:

- L2-normalized projected embeddings;
- positive bounded temperature, fixed or learnable by explicit config;
- numerically stable mixed-precision behavior;
- clear incomplete-biochemistry handling;
- patient identity is never a model feature;
- pairing cannot be broken by independent shuffling;
- validation retrieval metrics such as CT-to-bio and bio-to-CT Recall@K,
  alongside validation loss.

In-batch negatives are acceptable initially. Record actual batch size. Gradient
accumulation does not enlarge one InfoNCE negative set; do not claim it does.
Add a memory queue only if a demonstrated small-batch limitation justifies it.

Standard contrastive learning emphasizes shared CT-biochemistry information.
Preserve the pre-projection CT feature for transfer and never transfer
`ct_projector` to forecasting. CT-only self-supervision and factorized
shared/unique representations are future experiments.

## Same-timepoint regression control

Make it straightforward to compare contrastive pretraining against direct
CT-to-biochemistry regression because the downstream targets are continuous.
Prefer a task head/objective selected by configuration over duplicated encoders
or Datasets. For regression:

- input is CT only;
- output follows canonical biomarker order;
- loss is computed only where `bio_mask == 1`;
- use train-normalized targets and robust masked regression loss;
- save the same CT encoder artifact contract.

Implement contrastive-only first, regression second, and joint objectives only
after both work independently.

## Configuration

Add a version-controlled pretraining configuration containing at least:

```text
CT representation and preprocessing version
encoder name and base width
CT feature and projection dimensions
biochemistry hidden/feature/projection dimensions
temperature policy
batch size and epochs
learning rates / optimizer groups
weight decay
AMP and gradient clipping
seed
early-stopping metric and patience
```

Patient-data paths may be CLI arguments instead of committed YAML values. Save a
fully resolved configuration in every run directory. Do not introduce Hydra,
MONAI, Lightning, MLflow, or another large framework solely for this work.

## Pretraining checkpoint contract

Pretraining checkpoints must differ clearly from forecast checkpoints. Save at
least:

```python
{
    "checkpoint_type": "ct_bio_contrastive_pretraining",
    "checkpoint_version": 1,
    "ct_encoder_state": ...,
    "bio_encoder_state": ...,
    "ct_projector_state": ...,
    "bio_projector_state": ...,
    "model_config": ...,
    "ct_config": ...,
    "biomarker_order": ...,
    "normalization_stats": ...,
    "pretraining_split_provenance": ...,
    "optimizer_state": ...,
    "scaler_state": ...,
    "epoch": ...,
    "best_validation_metric": ...,
    "history": ...,
    "rng_state": ...,
}
```

Use atomic writes. Validate required keys/types. Support resuming pretraining
independently from initializing downstream training.

## Downstream transfer contract

Add one explicit downstream initialization option, for example:

```text
--ct-pretrained-checkpoint PATH
```

Its meaning is: start a new longitudinal experiment and initialize only the CT
encoder from a compatible pretraining checkpoint.

It must not load:

- contrastive projectors;
- pretraining biochemistry encoder by default;
- pretraining optimizer/scaler state;
- pretraining epoch/early-stopping state;
- forecast fusion/output head.

Keep it distinct from `--resume`, which resumes a complete longitudinal run.
Before transfer, validate checkpoint type/version, encoder architecture,
representation, expected shape, preprocessing version, feature dimensions, and
all state-dict keys/tensor shapes. Default to strict loading. Any compatibility
mapping must be explicit, tested, and reported.

Support downstream policies without duplicating training loops:

```text
frozen encoder -> train prediction modules
last stage trainable -> smaller CT learning rate
full encoder trainable -> smallest CT learning rate, experimental
```

The first transfer experiment should freeze the pretrained CT encoder. Later,
unfreeze the last ResNet stage with a smaller learning rate than new downstream
modules.

## Testing and acceptance

Tests use synthetic inputs only and require no patient data, network, or
downloaded weights.

### 3D ResNet

- production-contract input returns `[B,embedding_dim]`;
- forward/backward produces finite gradients;
- invalid rank/channel/dtype/range/non-finite values fail clearly;
- frozen parameters and normalization statistics stay frozen;
- save/load is deterministic;
- existing Conv3D, 2.5D, and legacy checkpoint tests still pass.

Use smaller synthetic volumes for routine test speed when appropriate, with one
separate contract test for configured production shape.

### Paired Dataset

- exact CT shape and canonical biomarker order;
- correct missing values/masks;
- train-only normalization;
- representation-aware paths/statuses;
- enforced duplicate-patient policy;
- rejection of downstream validation/test overlap;
- finite model tensors.

### Contrastive objective

- matched pairs use diagonal labels;
- similarity matrix is `[B,B]`;
- symmetric loss is the mean of both directions;
- projected embeddings are normalized;
- temperature remains valid;
- one optimization step changes trainable parameters;
- validation cannot shuffle away correspondence.

### Checkpoint and transfer

- pretraining checkpoint round-trip restores all pretraining modules;
- downstream transfer loads only CT encoder;
- forecast head and downstream bio encoder remain newly initialized;
- projector keys are excluded;
- incompatible encoder/representation/shape/version fails;
- `--resume` is unchanged;
- old forecast checkpoints still load.

### Synthetic integration

Run:

```text
paired Dataset -> DataLoader -> CT/Bio encoders -> projectors
-> contrastive loss -> backward -> optimizer step -> checkpoint
-> new forecast model -> CT-only transfer -> forecast forward/backward
```

For real data, stop after preflight and a small dry run until the user requests
cohort-scale training:

1. inspect 3-5 pairs;
2. validate Dataset and patient overlap;
3. run one CPU/GPU batch forward/backward;
4. run a short training/checkpoint round-trip;
5. report loss, retrieval metrics, memory, and runtime;
6. only then launch full pretraining.

## Recommended implementation sequence

Keep commits small and reviewable:

```text
1. chore(repo): preserve verified pre-contrastive recovery point
2. feat(ct): add configurable 3d resnet18 encoder
3. test(ct): cover 3d resnet contracts and legacy paths
4. feat(pretrain): add same-timepoint paired dataset and preflight
5. feat(pretrain): add ct-biochemistry contrastive model and loss
6. feat(pretrain): add training entry point, metrics, and checkpoints
7. feat(train): load compatible pretrained ct encoder for forecasting
8. test(pretrain): cover checkpoint transfer and synthetic integration
9. docs(pretrain): document boundaries and experiment commands
```

Do not combine everything into one opaque commit. Do not push or create a pull
request unless the user asks.

## Completion criteria

Complete only when:

- original 2.5D remains runnable;
- lightweight Conv3D remains selectable;
- `resnet18_3d` works through the existing factory/config boundary;
- the paired Dataset works without future targets;
- patient-level leakage checks are explicit and tested;
- both encoders train jointly under symmetric contrastive loss;
- projectors are excluded from downstream transfer;
- pretraining checkpoints preserve architecture, preprocessing, split, and
  normalization provenance;
- downstream `train.py` initializes only CT encoder from compatible weights;
- `--resume` retains its meaning;
- synthetic unit/integration tests pass;
- small real-data preflight reports counts/overlap without scaling up;
- final report lists changed files, commands, tests, unresolved assumptions,
  and final `git status` accurately.

## Non-goals for the first implementation

Do not add these without a demonstrated limitation and expanded user scope:

- full-volume variable-depth CT training;
- segmentation-derived three-channel CT inputs;
- MoCo queues or distributed contrastive training;
- factorized shared/unique latent spaces;
- masked CT reconstruction or a large multi-task objective;
- transfer of the pretraining biochemistry encoder by default;
- synthetic future-biochemistry labels;
- automatic cohort-wide training/preprocessing;
- remote pushes, tags, releases, or pull requests.
