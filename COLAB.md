# Training on Google Colab

Two runtime options, per the decision in `CLAUDE.md` section 3:

- **GPU runtime (recommended, use for the real submission)** — Colab gives you an NVIDIA
  T4/L4/A100 depending on tier. `scripts/colab_setup.sh` builds the official
  `diff_gaussian_rasterization` CUDA extension and `train.py` uses the exact tiled rasterizer
  from `graphdeco-inria/gaussian-splatting`.
- **CPU runtime** — no CUDA extension is built; `gaussian_renderer/cpu_rasterizer.py` (a
  from-scratch pure-PyTorch reference implementation, see its docstring) is used instead. It
  is **O(N_gaussians × H × W) per image with no tiling**, so it is only realistic for small
  smoke tests (a few thousand points, heavily downscaled renders) — not a competitive full
  training run. Use it to verify the pipeline runs end-to-end, then switch to GPU for real
  training (`Runtime > Change runtime type > GPU`).

## 1. Get the repo + data onto the Colab instance

```python
!git clone <your-repo-url> VAR-2026-BTS-Digital-Twin
%cd VAR-2026-BTS-Digital-Twin
# dataset/ is .gitignore'd (see .gitignore) -- upload/unzip it separately, e.g. from Drive:
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/<path-to>/dataset" .
```

## 2. Setup

```python
!bash scripts/colab_setup.sh
```

This installs the project (`pip install -e .`) and, only if `torch.cuda.is_available()`,
builds `diff_gaussian_rasterization`. Check the printed output to confirm which path was
taken before trusting training speed/quality.

## 3. Train

Single scene (useful for a first smoke test):

```python
!python src/train.py \
    --source_path dataset/phase1/private_set1/HCM0249 \
    --model_path output/private_set1/HCM0249 \
    --data_device cuda \
    --iterations 30000
```

All scenes in a set (private_set1 for submission, public_set for validation since it ships
ground-truth test images — see `CLAUDE.md` section 2):

```python
!python scripts/train_all.py \
    --dataset_root dataset/phase1/private_set1 \
    --output_root output/private_set1 \
    --data_device cuda

!python scripts/train_all.py \
    --dataset_root dataset/phase1/public_set \
    --output_root output/public_set \
    --data_device cuda
```

On a CPU runtime, drop to a handful of iterations and a single scene first
(`--iterations 500 --scenes HCM0181`) to confirm the loop runs without erroring before
attempting anything larger — see the CPU rasterizer's complexity warning above.

## 4. Render submission

```python
!python src/render_submission.py \
    --dataset_root dataset/phase1/private_set1 \
    --models_root output/private_set1 \
    --output_root submission/private_set1 \
    --data_device cuda
```

Produces `submission/private_set1/<scene>/0001.png, 0002.png, ... + index_map.json` — see
`CLAUDE.md` section 5 for the naming-convention assumption this makes (unconfirmed against
the actual submission spec, which wasn't fully provided).

## 5. Validate against public_set (has real ground truth)

```python
!python src/render_submission.py \
    --dataset_root dataset/phase1/public_set \
    --models_root output/public_set \
    --output_root submission/public_set \
    --data_device cuda

!python src/evaluate.py \
    --renders_root submission/public_set \
    --dataset_root dataset/phase1/public_set \
    --psnr_max 50 \
    --report_path eval_report.json
```

`--psnr_max 50` is a placeholder (see `CLAUDE.md` section 5 / `src/evaluate.py` docstring) —
replace with the organizer's official value once known.

## 6. Persist results

Colab instances are ephemeral — copy `output/` (checkpoints) and `submission/` back to Drive
before the runtime recycles:

```python
!cp -r output submission "/content/drive/MyDrive/<path-to>/"
```
