<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/motioninsight-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/motioninsight-light.svg">
    <img src="assets/motioninsight-light.svg" alt="MotionInsight — object motion diagnosis" width="100%">
  </picture>
</p>

<h1 align="center">MotionInsight: Diagnosing Object Motion<br>Deficiencies in Generated Videos</h1>

<p align="center"><strong>EMNLP 2026 Findings</strong></p>

<p align="center">
  Jiahao Zhan<sup>1,2</sup>, Yongrui Ma<sup>1,2</sup>, Qunliang Xing<sup>2</sup>, Xuanyu Zhang<sup>4</sup>,<br>
  Jingqi Tong<sup>3</sup>, Junlin Li<sup>2</sup>, Li Zhang<sup>2</sup>, Shijie Zhao<sup>2,†,✉</sup>, Tianfan Xue<sup>1,5,✉</sup>
</p>
<p align="center">
  <sup>1</sup>MMLab, CUHK &nbsp; <sup>2</sup>ByteDance Inc. &nbsp; <sup>3</sup>Fudan University<br>
  <sup>4</sup>Peking University &nbsp; <sup>5</sup>CPII under InnoHK<br>
  <sub>† Project Lead &nbsp; ✉ Corresponding Authors</sub>
</p>

<p align="center">
  <a href="paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-d95757?style=for-the-badge" alt="Paper PDF"></a>
  <a href="https://huggingface.co/JohnZhan/MotionInsight-8B"><img src="https://img.shields.io/badge/Hugging_Face-Models-e6aa32?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face Models"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Get_Started-Installation-138a7b?style=for-the-badge" alt="Installation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-526b81?style=for-the-badge" alt="Apache 2.0 License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#extract-motion-features">Preprocessing</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#training">Training</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

**From observing frames to diagnosing motion.** MotionInsight evaluates a designated
object in a generated video, explaining where its motion fails and scoring three
complementary dimensions of motion fidelity.

<p align="center">
  <img src="assets/motion-failures.png" width="100%" alt="Motion failures from the paper: a volleyball rises without contact, a tennis ball teleports, and a bumper car changes orientation inconsistently.">
</p>
<p align="center"><sub>Visually plausible frames can hide implausible motion. Examples from Figure 1 of the paper.</sub></p>

MotionInsight is an object-centric diagnostic evaluator for generated-video motion. It combines sampled RGB frames with explicit object-motion features from SAM3 and CoTracker3 and camera poses from VIPE. The released Qwen3-VL-8B checkpoint produces diagnostic reasoning and three continuous scores from **1.0 (poor)** to **5.0 (excellent)**.

| Dimension | What does MotionInsight diagnose? |
|---|---|
| **Object consistency** | Changes in identity, appearance, or structure during motion. |
| **Motion continuity** | Abrupt jumps, jitter, and discontinuous trajectories. |
| **Physical plausibility** | Motion that conflicts with forces, contact, or basic physical dynamics. |

The released checkpoint uses legacy JSON field names from training. `structural_stability` corresponds to object consistency, and `motion_coherence` corresponds to motion continuity.

This repository does **not** release VidMotion, its videos, or annotations. It contains inference, training, motion-feature extraction, and pinned third-party submodules. Model weights are hosted on [Hugging Face](https://huggingface.co/JohnZhan/MotionInsight-8B).

### Method

MotionInsight combines sampled RGB observations with explicit target-object tracks
and camera motion. Motion-description alignment connects the motion representations
to the VLM, followed by motion-specific GRPO for diagnostic assessment.

<p align="center">
  <img src="assets/method-overview.png" width="100%" alt="Paper method overview: tracking features and camera poses are encoded into motion embeddings and interleaved with visual tokens for Qwen3-VL.">
</p>
<p align="center"><sub>Method overview from Figure 2. See <a href="TRAINING.md">the training guide</a> for the released implementation.</sub></p>

## Repository layout

```text
MotionInsight/
├── assets/                             # logo and paper figures
├── docs/INSTALL.md                     # two-environment setup
├── requirements/                       # preprocessing, inference, training
├── checkpoints/                        # download MotionInsight-8B here
├── motion_features/
│   ├── extract_object_motion.py        # SAM3 + CoTracker3
│   └── extract_camera_motion.py        # VIPE
├── patches/modeling_qwen3_vl.py        # motion-aware Qwen3-VL implementation
├── scripts/
│   ├── patch_transformers.py
│   └── verify_checkpoint.py
├── training/                           # alignment and multi-task GRPO
│   ├── configs/                        # DeepSpeed ZeRO-2/ZeRO-3
│   ├── trainer/                        # motion-aware GRPO trainer
│   ├── run_alignment.sh
│   └── run_grpo.sh
├── thirdparty/
│   ├── co-tracker/
│   ├── sam3/
│   └── vipe/
├── inference.py
└── requirements.txt
```

The checkpoint is hosted separately on Hugging Face. It contains inference
artifacts only; optimizer shards, RNG state, scheduler state, trainer state, and
training arguments are excluded.

## Installation

Use **separate Python 3.10 environments** for preprocessing and training/inference.
The dependency files are based on the two supplied environments, with public
replacements for internal packages and a documented Hub-version correction.

| Environment | PyTorch / CUDA | Transformers | Requirements |
|---|---|---|---|
| Preprocessing | 2.7.0 / 12.8 | 4.48.3 | [preprocess.txt](requirements/preprocess.txt) |
| Training / inference | 2.11.0 / 12.6 | 4.57.1 + motion patch | [train.txt](requirements/train.txt) |
| Inference only | 2.11.0 / 12.6 | 4.57.1 + motion patch | [inference.txt](requirements/inference.txt) |

**Start with [the installation guide](docs/INSTALL.md)** for both environments,
CUDA extension builds, and attention-backend options. No virtual environments,
compiled extensions, extracted data, or model weights are committed to GitHub.

For inference on already extracted features:

```bash
git clone --recurse-submodules https://github.com/JohnZhan2023/MotionInsight.git
cd MotionInsight
python3.10 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/inference.txt
```

### Download the checkpoint

Download the released `multidim-all/checkpoint-5500` from
[JohnZhan/MotionInsight-8B](https://huggingface.co/JohnZhan/MotionInsight-8B):

```bash
hf download JohnZhan/MotionInsight-8B \
  --local-dir checkpoints/MotionInsight-8B
```

The resulting `checkpoints/MotionInsight-8B` directory is the default model path
used by `inference.py`.

The checkpoint stores approximately 17.7 GB of BF16 weights. At least 24 GB of GPU memory may work for short videos; 32 GB or more is recommended.

### Install the motion-aware Qwen3-VL implementation

MotionInsight adds object-motion and camera-motion projectors to Qwen3-VL. Install the included implementation into the active `transformers==4.57.1` environment before inference:

```bash
python scripts/patch_transformers.py
python scripts/patch_transformers.py --check
```

The command backs up the upstream module before replacing it. Restore it with:

```bash
python scripts/patch_transformers.py --restore
```

Restart existing Python processes after installing or restoring the patch.

## Extract motion features

Run these commands in `.venv-preprocess` after completing [preprocessing setup](docs/INSTALL.md#1-preprocessing).

The third-party model weights are not bundled. Download the SAM3 checkpoint after accepting its access terms, and download the CoTracker3 scaled offline checkpoint following the respective third-party READMEs:

- [SAM3 checkpoint instructions](thirdparty/sam3/README.md)
- [CoTracker3 checkpoint instructions](thirdparty/co-tracker/README.md)
- [VIPE setup and pretrained dependencies](thirdparty/vipe/README.md)

Extract target-object motion:

```bash
python motion_features/extract_object_motion.py \
  --video /path/to/video.mp4 \
  --target "speed bag" \
  --sam-checkpoint /path/to/sam3.pt \
  --cotracker-checkpoint /path/to/scaled_offline.pth \
  --output outputs/video_speed_bag.pt
```

Extract camera motion:

```bash
python motion_features/extract_camera_motion.py \
  --video /path/to/video.mp4 \
  --output outputs/video_camera.pt
```

Object feature files contain `x: [1, N, T, 1110]` and `confidence: [1, T, N]`. Camera files contain one `[4, 4]` pose matrix per frame. Do not alter the video timeline after extracting these features.

## Inference

Activate `.venv-model` before running inference.

Validate the release checkpoint:

```bash
python scripts/verify_checkpoint.py checkpoints/MotionInsight-8B
```

Run a single video:

```bash
python inference.py \
  --model_name_or_path checkpoints/MotionInsight-8B \
  --video_path /path/to/video.mp4 \
  --object_motion_path outputs/video_speed_bag.pt \
  --camera_motion_path outputs/video_camera.pt \
  --target "speed bag" \
  --output_path outputs/prediction.jsonl
```

The expected answer is:

```text
<thinking>diagnostic reasoning</thinking>
<answer>{"structural_stability": 3.0, "physical_plausibility": 4.5, "motion_coherence": 3.5}</answer>
```

For batch inference, pass a JSONL manifest:

```json
{"video_path":"/path/to/video.mp4","object_motion":"/path/to/object.pt","camera_motion":"/path/to/camera.pt","target":"speed bag","problem_type":"regression","data_type":"video","question":"Give the target object speed bag three motion-quality scores."}
```

```bash
python inference.py \
  --model_name_or_path checkpoints/MotionInsight-8B \
  --dataset_path /path/to/inputs.jsonl \
  --output_path outputs/predictions.jsonl
```

The released setting uses `--frame_stride 16` and deterministic decoding.

## Training

Install [training dependencies](docs/INSTALL.md#2-training-and-inference) in `.venv-model`.
The launchers default to FlashAttention 2; set `ATTN_IMPLEMENTATION=sdpa` to use
PyTorch attention without installing a separate FlashAttention extension.

The two released training stages follow the paper setting: modality alignment for
3 epochs and multi-task GRPO for 5 epochs on 8 NVIDIA A100 GPUs, both at learning
rate `1e-6`. GRPO uses 8 sampled responses, KL coefficient `0.001`, and asymmetric
overestimation penalty `1.5`.

```bash
bash training/run_alignment.sh \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/alignment.jsonl \
  outputs/motioninsight-aligned

bash training/run_grpo.sh \
  outputs/motioninsight-aligned \
  /path/to/multitask.jsonl \
  outputs/motioninsight-grpo
```

See [TRAINING.md](TRAINING.md) for the complete setting and annotation schemas.
The training data itself is not distributed.

## Model scope

MotionInsight targets entities with clear spatial boundaries, persistent identities, and trackable trajectories. It is less suitable for fluids, smoke, fire, splashes, or highly deformable materials. Scores can also be affected by target ambiguity, segmentation/tracking failures, camera-pose errors, temporal misalignment, and out-of-distribution content.

## Third-party licenses

The root Apache-2.0 license applies only where indicated. Vendored dependencies retain their original licenses:

- CoTracker: CC BY-NC 4.0.
- SAM3: SAM License.
- VIPE: Apache-2.0, with additional notices in its `THIRD_PARTY_LICENSES.md`.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution. In particular, CoTracker's license restricts commercial use.

## Citation

```bibtex
@inproceedings{zhan2026motioninsight,
  title  = {MotionInsight: Diagnosing Object Motion Deficiencies in Generated Videos},
  author = {Zhan, Jiahao and Ma, Yongrui and Xing, Qunliang and Zhang, Xuanyu and Tong, Jingqi and Li, Junlin and Li, Zhang and Zhao, Shijie and Xue, Tianfan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year   = {2026}
}
```

## License

MotionInsight code and checkpoint files in this release are provided under the [Apache License 2.0](LICENSE), except for components under `thirdparty/`, which retain their own license terms.
