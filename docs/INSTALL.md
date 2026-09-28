# Installation

Use **two separate Python 3.10 environments**: one for motion-feature extraction,
and one for training and inference. The requirements are curated from the supplied
working directories, with explicit public-package replacements described below.
They are not a full `pip freeze` of either environment.

| Component | Preprocessing | Training / inference |
|---|---|---|
| Source Python | 3.10.19 | 3.10.9 |
| PyTorch | 2.7.0 + CUDA 12.8 | 2.11.0 + CUDA 12.6 |
| torchvision | 0.22.0 | 0.26.0 |
| Transformers | 4.48.3 | 4.57.1 + MotionInsight patch |
| NumPy | 1.26.4 | 1.26.4 |
| Requirements | `requirements/preprocess.txt` | `requirements/train.txt` |

The target platform is Linux with an NVIDIA GPU. Building VIPE requires a CUDA
toolkit with `nvcc` compatible with the installed PyTorch build, a C++ compiler,
and the system libraries described in its [environment file](../thirdparty/vipe/envs/base.yml).
Installing a PyTorch wheel does not install the `nvcc` compiler.

## Clone

```bash
git clone --recurse-submodules https://github.com/JohnZhan2023/MotionInsight.git
cd MotionInsight
# Also works for an existing clone:
git submodule update --init --recursive
```

Use the pinned submodules: the CoTracker and SAM3 forks contain compatibility
changes and expose the features consumed by this repository.

## 1. Preprocessing

```bash
python3.10 -m venv .venv-preprocess
source .venv-preprocess/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/preprocess.txt

python -m pip install -c requirements/preprocess.txt -e thirdparty/co-tracker
python -m pip install -c requirements/preprocess.txt -e thirdparty/sam3
python -m pip install -c requirements/preprocess.txt --no-build-isolation -e thirdparty/vipe
python -m pip check
```

Run the two commands in [Extract motion features](../README.md#extract-motion-features)
in this environment, then `deactivate`. SAM3, CoTracker3, and VIPE model weights
must be downloaded separately following their respective READMEs. SAM3 access
may require accepting its model terms.

The source environment recorded `huggingface-hub==0.25.1`, which conflicts with
the declared requirements of `transformers==4.48.3`. The public requirements use
`huggingface-hub==0.36.2` to satisfy that dependency. NumPy remains pinned below 2
as required by the SAM3 fork; OpenCV is pinned to avoid upgrading NumPy.

## 2. Training and inference

```bash
python3.10 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts/patch_transformers.py
python scripts/patch_transformers.py --check
python -m pip check
```

For inference only, replace `requirements.txt` with
`requirements/inference.txt` to omit DeepSpeed, TRL, and training integrations.

The public training requirements use `wandb==0.19.11` in place of the source
environment's internal W&B distribution. The launchers use `--report_to none`;
an online logging account is not required. Internal service clients, unrelated
inference engines, local editable paths, caches, and CUDA binary artifacts from
the original environments are intentionally excluded.

### Attention backend

Inference uses the model's default attention backend and does not require
FlashAttention. The training launchers default to `flash_attention_2` to preserve
the original setting. The inspected source environment has
`flash-attn==2.7.4.post1`, but its installed extension fails to import against
PyTorch 2.11.0 with an undefined CUDA symbol. Copying that extension or freezing
its version does not reproduce a usable installation.

For a fresh environment without FlashAttention, select PyTorch SDPA explicitly:

```bash
ATTN_IMPLEMENTATION=sdpa bash training/run_alignment.sh \
  /path/to/Qwen3-VL-8B-Instruct /path/to/alignment.jsonl outputs/aligned

ATTN_IMPLEMENTATION=sdpa bash training/run_grpo.sh \
  outputs/aligned /path/to/multitask.jsonl outputs/grpo
```

SDPA is an alternate execution backend; its runtime and memory use can differ
from the paper's FlashAttention setting. To use `flash_attention_2`, install a
FlashAttention build compatible with your exact PyTorch, CUDA toolkit, and Python
versions, following the [upstream installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).
Check `python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"`
before launching training. Do not leave an incompatible FlashAttention extension
installed in an otherwise clean SDPA environment.

## Environment boundaries

Only the extracted `.pt` features and the original video are shared between the
two environments. Run `scripts/patch_transformers.py` in `.venv-model` only.
Reinstalling Transformers replaces the patch, so rerun it after such changes.
The helper creates a backup and supports `--restore`.

These files document the observed versions and resolve public dependencies;
they do not certify a fresh end-to-end GPU training run. VIPE compilation,
third-party weight access, and a complete preprocessing/inference run still
depend on the target machine and downloaded models.
