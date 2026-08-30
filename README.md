# MotionInsight: Diagnosing Object Motion Deficiencies in Generated Videos

**Jiahao Zhan, Yongrui Ma, Qunliang Xing, Xuanyu Zhang, Jingqi Tong, Junlin Li, Li Zhang, Shijie Zhao, Tianfan Xue**

[Paper](paper.pdf)

MotionInsight is an object-centric diagnostic evaluator for generated-video motion. It combines sampled RGB frames with explicit object-motion features from SAM3 and CoTracker3 and camera poses from VIPE. The released Qwen3-VL-8B checkpoint produces diagnostic reasoning and three continuous scores from **1.0 (poor)** to **5.0 (excellent)**.

The paper evaluates:

- **Object consistency**: stable identity, appearance, and structure during motion.
- **Motion continuity**: smooth, continuous trajectories without abrupt jumps or jitter.
- **Physical plausibility**: motion consistent with forces and basic physical dynamics.

The released checkpoint uses legacy JSON field names from training. `structural_stability` corresponds to object consistency, and `motion_coherence` corresponds to motion continuity.

This repository does **not** release VidMotion, its videos, or annotations. It contains model inference, paper-setting training code, motion-feature extraction, the checkpoint, and required third-party source trees.

## Repository layout

```text
MotionInsight-OpenSource/
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

Python 3.12 or later and CUDA are recommended for the complete extraction and inference pipeline, matching SAM3's documented environment.

```bash
git clone --recurse-submodules https://github.com/JohnZhan2023/MotionInsight.git
cd MotionInsight

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Install the vendored feature extractors and their dependencies.
pip install -e thirdparty/co-tracker
pip install -e thirdparty/sam3
pip install -e thirdparty/vipe
```

For an existing clone, initialize the dependencies with:

```bash
git submodule update --init --recursive
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

The checkpoint stores approximately 18.7 GB of BF16 weights. At least 24 GB of GPU memory may work for short videos; 32 GB or more is recommended.

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
@misc{zhan2026motioninsight,
  title  = {MotionInsight: Diagnosing Object Motion Deficiencies in Generated Videos},
  author = {Zhan, Jiahao and Ma, Yongrui and Xing, Qunliang and Zhang, Xuanyu and Tong, Jingqi and Li, Junlin and Li, Zhang and Zhao, Shijie and Xue, Tianfan},
  year   = {2026}
}
```

## License

MotionInsight code and checkpoint files in this release are provided under the [Apache License 2.0](LICENSE), except for components under `thirdparty/`, which retain their own license terms.
