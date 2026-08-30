# Training MotionInsight

The released checkpoint is trained in two stages. This repository includes the
training implementation and paper hyperparameters, but does not include VidMotion,
video files, extracted features, or annotations.

## Paper setting

| Stage | Trainable parameters | Hardware | Epochs | Learning rate | Other settings |
|---|---|---:|---:|---:|---|
| Modality alignment | Motion projectors and fusion module | 8 x NVIDIA A100 | 3 | `1e-6` | BF16, cosine schedule, frame stride 16 |
| Multi-task GRPO | Full aligned model | 8 x NVIDIA A100 | 5 | `1e-6` | 8 generations, KL coefficient `0.001`, asymmetric overestimation penalty `1.5` |

The launchers use per-device batch size 1, gradient accumulation 1, gradient
checkpointing, FlashAttention 2, AdamW weight decay `0.01`, and maximum gradient
norm 5. The GRPO prompt and completion limits are 8192 and 384 tokens.

The implementation is checkpoint-faithful: object tracks are confidence-weighted
into 1110-dimensional frame features, camera rotations are represented in 9D,
and both are averaged over 16-frame intervals before the motion projectors and
fusion module. These modules correspond exactly to the motion-related parameter
names present in `checkpoints/MotionInsight-8B`.

## Annotation formats

All paths may be absolute or relative to the process working directory. Motion
features must be produced by the scripts in `motion_features/` from the same,
unaltered video timeline.

The alignment JSONL contains one object per line:

```json
{"video_path":"/data/example.mp4","object_motion":"/features/example_object.pt","camera_motion":"/features/example_camera.pt","target":"example object","data_type":"video","solution":"A concise description of the target object's motion."}
```

The multi-task GRPO JSONL uses this common schema:

```json
{"path":"/data/example.mp4","object_motion":"/features/example_object.pt","camera_motion":"/features/example_camera.pt","target":"example object","data_type":"video","problem_type":"regression","problem":"Evaluate the target object's motion quality.","solution":"<thinking>Reference reasoning.</thinking><answer>{\"structural_stability\": 3.0, \"physical_plausibility\": 4.0, \"motion_coherence\": 3.5}</answer>"}
```

Supported `problem_type` values are:

- `regression`: `solution` contains all three legacy score keys in the 1-5 range.
- `multiple choice`: add an `options` array and put the answer letter inside the answer tags.
- `discrimination`: the answer is `Real` or `Fake`.
- `cause-inspection`: the trainer also supports the issue-label JSON format used by the reward implementation.

The legacy names `structural_stability` and `motion_coherence` map to Object
Consistency and Motion Continuity in the paper.

## Environment

Install the repository dependencies and the motion-aware Qwen3-VL module:

```bash
pip install -r requirements.txt
python scripts/patch_transformers.py
python scripts/patch_transformers.py --check
```

Training expects a Qwen3-VL-8B-Instruct-compatible base model. The alignment
launcher adds `<|motion|>` to the tokenizer, freezes the VLM backbone, and trains
only `motion_projector`, `camera_motion_projector`, and `motion_fusion`.

## Stage 1: modality alignment

```bash
bash training/run_alignment.sh \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/alignment.jsonl \
  outputs/motioninsight-aligned
```

## Stage 2: multi-task GRPO

```bash
bash training/run_grpo.sh \
  outputs/motioninsight-aligned \
  /path/to/multitask.jsonl \
  outputs/motioninsight-grpo
```

The GRPO accuracy reward combines task-specific correctness with the paper's
asymmetric regression loss. For each regression dimension, overestimation is
penalized by a factor of `1.5`; the three dimension rewards are multiplied.

Both launchers assume one node with eight visible GPUs. Override
`CUDA_VISIBLE_DEVICES`, `MASTER_ADDR`, or `MASTER_PORT` through environment
variables when integrating them with a cluster launcher. Changing the number of
processes requires editing `--nproc_per_node` and revalidating the effective batch
and generation grouping.
