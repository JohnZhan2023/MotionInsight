# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

MOTION_TOKEN = "<|motion|>"
DEFAULT_NUM_MOTION_TOKENS = 16
DEFAULT_FRAME_STRIDE = 16
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., "
    "<thinking> reasoning process here </thinking><answer> answer here </answer>"
)
QUESTION_TEMPLATE = "{Question}\n"
TYPE_TEMPLATE = {
    "regression": " Please provide the thinking process within the <thinking> </thinking> tags and output only the three scores within the <answer> </answer> tags, one decimal place each, using this exact format: {\"structural_stability\": 3.2, \"motion_coherence\": 2.8, \"physical_plausibility\": 3.5}. Be strict and conservative in scoring. A mostly static object is not necessarily poor.",
    "discrimination": "Please provide the thinking process within the <thinking> </thinking> tags and answer only Real or Fake within the <answer> </answer> tags.",
    "cause-inspection": "Please provide the thinking process within the <thinking> </thinking> tags and provide a strict JSON object in <answer> </answer> as {\"issues\": [\"interpenetration\", \"split_or_merge\"]}. Allowed labels are: interpenetration, split_or_merge, pop_in_or_disappear, deformation, texture_flicker, stiff_movement, teleportation, gravity_violation, causality_error, secondary_motion_error, buoyancy_error, energy_conservation_violation. Use {\"issues\": []} when there is no issue.",
}

REGRESSION_DIMENSION_ALIASES = {
    "structural_stability": {
        "structural_stability",
        "structural stability",
        "structual_stability",
        "structual stability",
    },
    "motion_coherence": {
        "motion_coherence",
        "motion coherence",
    },
    "physical_plausibility": {
        "physical_plausibility",
        "physical plausibility",
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_motion_features(motion_path: str) -> Dict[str, torch.Tensor]:
    try:
        motion_obj = torch.load(motion_path, map_location="cpu", weights_only=True)
    except TypeError:
        motion_obj = torch.load(motion_path, map_location="cpu")

    if isinstance(motion_obj, torch.Tensor):
        return {"frame_motion": motion_obj.float()}

    if "pose" in motion_obj:
        return {"frame_motion": motion_obj["pose"].float()}

    if "x" in motion_obj and "confidence" in motion_obj:
        return {
            "x": motion_obj["x"].float(),
            "confidence": motion_obj["confidence"].float(),
        }

    raise ValueError(
        f"Unsupported motion feature format in {motion_path}. Expected a Tensor or a dict with `x` and `confidence`."
    )


def aggregate_tracks_to_frames(motion_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "frame_motion" in motion_features:
        frame_motion = motion_features["frame_motion"]
        if frame_motion.ndim > 2:
            frame_motion = frame_motion.reshape(frame_motion.shape[0], -1)
        return frame_motion

    x = motion_features["x"]
    confidence = motion_features["confidence"]
    x_frame = x.squeeze(0).permute(1, 0, 2)
    track_conf = confidence.squeeze(0)
    track_weight = torch.softmax(track_conf, dim=-1)
    return torch.einsum("tn,tnc->tc", track_weight, x_frame)


def convert_camera_pose_to_9d(frame_motion: torch.Tensor) -> torch.Tensor:
    if frame_motion.ndim == 3 and frame_motion.shape[-2:] == (4, 4):
        return frame_motion[:, :3, :3].reshape(frame_motion.shape[0], 9)
    if frame_motion.ndim == 2 and frame_motion.shape[-1] == 16:
        return frame_motion.view(frame_motion.shape[0], 4, 4)[:, :3, :3].reshape(frame_motion.shape[0], 9)
    if frame_motion.ndim == 2 and frame_motion.shape[-1] == 9:
        return frame_motion
    raise ValueError(
        f"Unsupported camera motion shape {tuple(frame_motion.shape)}. Expected [T,4,4], [T,16], or [T,9]."
    )


def build_temporal_sampling_plan(total_frames: int, frame_stride: int) -> tuple[list[int], list[int], list[int]]:
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames, got {total_frames}.")

    image_indices = []
    motion_token_frame_indices = []
    motion_counts_after_images = []

    current_frame = 0
    while current_frame < total_frames:
        image_indices.append(current_frame)

        motion_start = current_frame + 1
        motion_end = min(current_frame + 1 + frame_stride, total_frames)
        current_motion_indices = list(range(motion_start, motion_end))
        motion_token_frame_indices.extend(current_motion_indices)
        motion_counts_after_images.append(len(current_motion_indices))

        current_frame += frame_stride + 1

    return image_indices, motion_token_frame_indices, motion_counts_after_images


def build_motion_token_values(
    frame_motion: torch.Tensor,
    motion_token_frame_indices: list[int],
) -> torch.Tensor:
    if frame_motion.ndim != 2:
        raise ValueError(f"Expected frame motion with shape [T, C], got {tuple(frame_motion.shape)}.")

    if not motion_token_frame_indices:
        return frame_motion.new_zeros((0, frame_motion.shape[-1]))

    if frame_motion.shape[0] == 0:
        return torch.zeros((len(motion_token_frame_indices), frame_motion.shape[-1]), dtype=frame_motion.dtype)

    clipped_indices = [min(idx, frame_motion.shape[0] - 1) for idx in motion_token_frame_indices]
    return frame_motion[clipped_indices]


def sample_video_frames(
    video_path: str,
    stride: int,
):
    try:
        import decord
        from PIL import Image

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video contains no frames: {video_path}")
        image_indices, motion_token_frame_indices, motion_counts_after_images = build_temporal_sampling_plan(
            total_frames=total_frames,
            frame_stride=stride,
        )
        frames = vr.get_batch(image_indices).asnumpy()
        images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    except Exception:
        import torchvision
        from torchvision.transforms import ToPILImage

        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")
        total_frames = int(video.shape[0])
        if total_frames <= 0:
            raise ValueError(f"Video contains no frames: {video_path}")
        image_indices, motion_token_frame_indices, motion_counts_after_images = build_temporal_sampling_plan(
            total_frames=total_frames,
            frame_stride=stride,
        )
        images = [ToPILImage()(video[idx]) for idx in image_indices]

    return images, image_indices, motion_token_frame_indices, motion_counts_after_images, total_frames


def build_prompt_from_example(example: Dict[str, Any]) -> str:
    problem = example.get("problem")
    problem_type = example.get("problem_type")
    if isinstance(problem, str) and problem.strip() and isinstance(problem_type, str):
        question = problem
        if str(problem_type).strip().lower() == "multiple choice":
            options = example.get("options", [])
            if isinstance(options, list) and options:
                question += "Options:\n"
                for op in options:
                    question += f"{op}\n"
        return question.strip()

    for key in ("question", "prompt", "Question", "instruction"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    target = example.get("target", "object")
    return f"Describe the motion of the {target} in this video."


def load_jsonl(path: str) -> list[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_answer_tag(text: str) -> str:
    if not isinstance(text, str):
        return ""
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return ""
    return text[start_idx + len(start_tag):end_idx].strip()


def normalize_number(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def canonicalize_regression_key(key: str) -> Optional[str]:
    normalized_key = str(key).strip().lower().replace("-", "_")
    normalized_key_space = normalized_key.replace("_", " ")
    for canonical_key, aliases in REGRESSION_DIMENSION_ALIASES.items():
        if normalized_key in aliases or normalized_key_space in aliases:
            return canonical_key
    return None


def parse_regression_scores(answer_text: str) -> Dict[str, float]:
    if not answer_text:
        return {}

    parsed_scores: Dict[str, float] = {}

    try:
        obj = json.loads(answer_text)
        if isinstance(obj, dict):
            for key, value in obj.items():
                canonical_key = canonicalize_regression_key(key)
                if canonical_key is None:
                    continue
                score = normalize_number(value)
                if score is not None:
                    parsed_scores[canonical_key] = score
    except Exception:
        pass

    if parsed_scores:
        return parsed_scores

    import re

    pattern = re.compile(
        r"(struct(?:ural|ual)\s+stability|motion\s+coherence|physical\s+plausibility)"
        r"\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for key, value in pattern.findall(answer_text):
        canonical_key = canonicalize_regression_key(key)
        score = normalize_number(value)
        if canonical_key is not None and score is not None:
            parsed_scores[canonical_key] = score

    return parsed_scores


def parse_prediction(example: Dict[str, Any], prediction: str) -> Optional[Dict[str, Any]]:
    answer_text = extract_answer_tag(prediction)
    if not answer_text:
        return None

    problem_type = str(example.get("problem_type", "")).strip().lower()
    if problem_type == "regression":
        scores = parse_regression_scores(answer_text)
        if scores:
            return scores
        return {"raw_answer": answer_text}

    return {"raw_answer": answer_text}


def resolve_video_path(example: Dict[str, Any]) -> str:
    prompt = example.get("prompt")
    if isinstance(prompt, list):
        for message in prompt:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "video":
                    for key in ("video", "video_path", "path"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            return value

    for key in ("video_path", "video", "video_file", "path"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Could not find video path in sample. Expected one of: video_path, video, video_file, path.")


def resolve_video_path_from_content_item(item: Dict[str, Any], example: Dict[str, Any]) -> str:
    for key in ("video", "video_path", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return resolve_video_path(example)


def resolve_motion_path(example: Dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def prepare_motion_inputs(
    example: Dict[str, Any],
    motion_token_frame_indices: list[int],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    object_motion_path = resolve_motion_path(
        example,
        ("object_motion", "object_motion_path", "foreground_motion", "fg_motion"),
    )
    camera_motion_path = resolve_motion_path(
        example,
        ("camera_motion", "camera_motion_path", "background_motion", "bg_motion"),
    )

    object_motion_segments = None
    if object_motion_path is not None:
        object_frame_motion = aggregate_tracks_to_frames(load_motion_features(object_motion_path))
        object_motion_segments = build_motion_token_values(object_frame_motion, motion_token_frame_indices)

    camera_motion_segments = None
    if camera_motion_path is not None:
        camera_frame_motion = aggregate_tracks_to_frames(load_motion_features(camera_motion_path))
        camera_frame_motion = convert_camera_pose_to_9d(camera_frame_motion)
        camera_motion_segments = build_motion_token_values(camera_frame_motion, motion_token_frame_indices)

    return object_motion_segments, camera_motion_segments


def build_video_content_block(
    sampled_images,
    sampled_frame_indices,
    motion_counts_after_images,
    total_frames: int,
    frame_stride: int,
):
    content = []
    motion_token_count = 0
    for frame_idx, frame_image in enumerate(sampled_images):
        content.append({"type": "image", "image": frame_image})
        for _ in range(motion_counts_after_images[frame_idx]):
            content.append({"type": "text", "text": MOTION_TOKEN})
            motion_token_count += 1

    meta = {
        "frame_stride": frame_stride,
        "sampled_frame_indices": sampled_frame_indices,
        "motion_tokens_per_gap": frame_stride,
        "motion_token_count": motion_token_count,
        "num_sampled_images": len(sampled_images),
        "total_video_frames": total_frames,
    }
    return content, meta


def normalize_prompt_messages(example: Dict[str, Any]) -> list[Dict[str, Any]]:
    question = build_prompt_from_example(example)
    problem_type = str(example.get("problem_type", "")).strip().lower()
    data_type = example.get("data_type", "video")
    task_suffix = TYPE_TEMPLATE.get(problem_type, "")
    user_text = SYSTEM_PROMPT + QUESTION_TEMPLATE.format(Question=question) + task_suffix

    return [
        {
            "role": "user",
            "content": [
                {"type": data_type},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def materialize_messages(
    example: Dict[str, Any],
    frame_stride: int,
    object_motion_segments: Optional[torch.Tensor],
    camera_motion_segments: Optional[torch.Tensor],
) -> tuple[list[Dict[str, Any]], Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
    messages = normalize_prompt_messages(example)
    inference_meta: Dict[str, Any] = {"frame_stride": frame_stride}
    all_motion_token_frame_indices: list[int] = []

    resolved_any_video = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        new_content = []
        for item in content:
            item_type = item.get("type")
            if item_type == "video":
                video_path = resolve_video_path_from_content_item(item, example)
                sampled_images, sampled_frame_indices, motion_token_frame_indices, motion_counts_after_images, total_frames = sample_video_frames(
                    video_path=video_path,
                    stride=frame_stride,
                )
                all_motion_token_frame_indices = motion_token_frame_indices
                if object_motion_segments is not None:
                    object_motion_segments = object_motion_segments[: len(motion_token_frame_indices)]
                if camera_motion_segments is not None:
                    camera_motion_segments = camera_motion_segments[: len(motion_token_frame_indices)]

                video_block, video_meta = build_video_content_block(
                    sampled_images=sampled_images,
                    sampled_frame_indices=sampled_frame_indices,
                    motion_counts_after_images=motion_counts_after_images,
                    total_frames=total_frames,
                    frame_stride=frame_stride,
                )
                new_content.extend(video_block)
                inference_meta.update({"resolved_video_path": video_path, **video_meta})
                resolved_any_video = True
            else:
                new_content.append(item)
        message["content"] = new_content

    if not resolved_any_video:
        video_path = resolve_video_path(example)
        sampled_images, sampled_frame_indices, motion_token_frame_indices, motion_counts_after_images, total_frames = sample_video_frames(
            video_path=video_path,
            stride=frame_stride,
        )
        if object_motion_segments is not None:
            object_motion_segments = object_motion_segments[: len(motion_token_frame_indices)]
        if camera_motion_segments is not None:
            camera_motion_segments = camera_motion_segments[: len(motion_token_frame_indices)]
        all_motion_token_frame_indices = motion_token_frame_indices

        video_block, video_meta = build_video_content_block(
            sampled_images=sampled_images,
            sampled_frame_indices=sampled_frame_indices,
            motion_counts_after_images=motion_counts_after_images,
            total_frames=total_frames,
            frame_stride=frame_stride,
        )
        inserted = False
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                message["content"] = video_block + message["content"]
                inserted = True
                break
        if not inserted:
            messages.append({"role": "user", "content": video_block})
        inference_meta.update({"resolved_video_path": video_path, **video_meta})

    inference_meta["motion_token_frame_indices"] = all_motion_token_frame_indices
    return messages, inference_meta, object_motion_segments, camera_motion_segments


def parse_args():
    parser = argparse.ArgumentParser("MotionInsight inference")

    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/MotionInsight-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset_path",
        type=str,
        help="Input jsonl file for batch inference.",
    )
    source.add_argument("--video_path", type=str, help="Input video for single-video inference.")
    parser.add_argument("--object_motion_path", type=str, help="CoTracker feature .pt for --video_path.")
    parser.add_argument("--camera_motion_path", type=str, help="VIPE pose .pt for --video_path.")
    parser.add_argument("--target", type=str, default="object", help="Target object for --video_path.")
    parser.add_argument("--question", type=str, default=None, help="Override the default scoring question.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/motioninsight_predictions.jsonl",
        help="Path to save jsonl predictions.",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Use one sampled frame, then insert one motion token per skipped frame before the next sampled frame.",
    )

    return parser.parse_args()


def build_model_inputs(
    processor,
    messages,
    device: str,
    object_motion_segments: Optional[torch.Tensor],
    camera_motion_segments: Optional[torch.Tensor],
):
    text = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    ]

    image_inputs, _, _ = process_vision_info([messages], return_video_kwargs=True)
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    if object_motion_segments is not None:
        inputs["motion_values"] = [object_motion_segments]
        inputs["object_motion_values"] = [object_motion_segments]
    if camera_motion_segments is not None:
        inputs["camera_motion_values"] = [camera_motion_segments]

    return inputs.to(device)


def generate_answer(
    model,
    processor,
    device: str,
    video_path: str,
    question: str,
    max_new_tokens: int,
    frame_stride: int,
    object_motion_segments: Optional[torch.Tensor] = None,
    camera_motion_segments: Optional[torch.Tensor] = None,
    example: Optional[Dict[str, Any]] = None,
):
    if example is None:
        example = {
            "video_path": video_path,
            "question": question,
        }
    else:
        example = dict(example)
        example.setdefault("video_path", video_path)
        if question is not None:
            example.setdefault("question", question)

    if object_motion_segments is None and camera_motion_segments is None:
        _, _, motion_token_frame_indices, _, _ = sample_video_frames(
            video_path=resolve_video_path(example),
            stride=frame_stride,
        )
        object_motion_segments, camera_motion_segments = prepare_motion_inputs(
            example=example,
            motion_token_frame_indices=motion_token_frame_indices,
        )

    messages, input_meta, object_motion_segments, camera_motion_segments = materialize_messages(
        example=example,
        frame_stride=frame_stride,
        object_motion_segments=object_motion_segments,
        camera_motion_segments=camera_motion_segments,
    )
    input_meta["object_motion_tokens"] = None if object_motion_segments is None else int(object_motion_segments.shape[0])
    input_meta["camera_motion_tokens"] = None if camera_motion_segments is None else int(camera_motion_segments.shape[0])
    inputs = build_model_inputs(processor, messages, device, object_motion_segments, camera_motion_segments)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer, input_meta


def run_dataset(args, model, processor):
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    samples = load_jsonl(args.dataset_path)
    if not samples:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")
    if args.start_index < 0 or args.start_index >= len(samples):
        raise IndexError(f"start_index {args.start_index} is out of range for dataset of size {len(samples)}")

    if args.num_samples is None:
        end_index = len(samples)
    else:
        end_index = min(len(samples), args.start_index + args.num_samples)

    selected_samples = samples[args.start_index:end_index]
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Running batch inference on {len(selected_samples)} samples from {args.dataset_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for offset, example in enumerate(selected_samples):
            sample_idx = args.start_index + offset
            record = dict(example)
            record["index"] = sample_idx

            try:
                question = build_prompt_from_example(example)
                video_path = resolve_video_path(example)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")

                answer, input_meta = generate_answer(
                    model=model,
                    processor=processor,
                    device=args.device,
                    video_path=video_path,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    frame_stride=args.frame_stride,
                    example=example,
                )

                record.update(
                    {
                        "resolved_video_path": video_path,
                        "resolved_question": question,
                        "prediction": answer,
                        "parsed_prediction": parse_prediction(example, answer),
                        "inference_meta": {
                            **input_meta,
                        },
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "prediction": None,
                        "error": str(exc),
                    }
                )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (offset + 1) % 10 == 0 or offset == len(selected_samples) - 1:
                print(f"Processed {offset + 1}/{len(selected_samples)} samples")

    print(f"Saved predictions to {args.output_path}")


def run_single(args, model, processor):
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not args.object_motion_path and not args.camera_motion_path:
        raise ValueError("Provide at least one of --object_motion_path or --camera_motion_path.")
    for label, path in (
        ("object motion", args.object_motion_path),
        ("camera motion", args.camera_motion_path),
    ):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{label.title()} file not found: {path}")

    question = args.question or (
        f'Give the target object "{args.target}" in this video three continuous scores from 1 to 5 for the following '
        "aspects in order: structural stability, physical plausibility, and motion coherence."
    )
    example = {
        "video_path": args.video_path,
        "object_motion": args.object_motion_path,
        "camera_motion": args.camera_motion_path,
        "target": args.target,
        "question": question,
        "problem_type": "regression",
        "data_type": "video",
    }
    answer, input_meta = generate_answer(
        model=model,
        processor=processor,
        device=args.device,
        video_path=args.video_path,
        question=question,
        max_new_tokens=args.max_new_tokens,
        frame_stride=args.frame_stride,
        example=example,
    )
    result = {
        **example,
        "prediction": answer,
        "parsed_prediction": parse_prediction(example, answer),
        "inference_meta": input_meta,
    }
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result["parsed_prediction"], ensure_ascii=False))
    print(f"Saved prediction to {args.output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading Qwen3-VL model from {args.model_name_or_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    num_added_tokens = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [MOTION_TOKEN]}
    )
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
    model.config.motion_token_id = processor.tokenizer.convert_tokens_to_ids(MOTION_TOKEN)
    model.eval()

    if args.dataset_path:
        run_dataset(args, model, processor)
    else:
        run_single(args, model, processor)


if __name__ == "__main__":
    main()
