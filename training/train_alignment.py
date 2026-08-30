# Copyright 2024. All rights reserved.
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
"""Stage-one modality alignment for MotionInsight."""

import random
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

MOTION_TOKEN = "<|motion|>"
MOTION_FEATURE_DIM = 1110
FRAME_STRIDE = 16
processor = None
motion_token_id = None
MOTION_PROMPT_TEMPLATES = [
    "Describe the motion of the {target} in this video.",
    "How does the {target} move in this video?",
    "What motion does the {target} exhibit in this video?",
    "Explain how the {target} is moving throughout the video.",
    "Summarize the movement of the {target} in this video.",
    "What is the {target} doing motion-wise in this video?",
    "Describe how the {target}'s motion changes over time in this video.",
    "Provide a brief description of the {target}'s movement in this video.",
]


def resolve_media_path(example: Dict[str, Any]) -> str:
    """Resolve the media path while supporting both the new and legacy dataset schemas."""
    media_path = example.get("video_path", example.get("path"))
    if media_path is None:
        raise ValueError("Each sample must provide either `video_path` or `path`.")
    return media_path


def load_motion_features(motion_path: str) -> Dict[str, torch.Tensor]:
    """Load raw motion features from disk without collapsing the track structure."""
    try:
        motion_obj = torch.load(motion_path, map_location="cpu", weights_only=True)
    except TypeError:
        motion_obj = torch.load(motion_path, map_location="cpu")

    if isinstance(motion_obj, torch.Tensor):
        return {"frame_motion": motion_obj.float()}

    if "x" in motion_obj and "confidence" in motion_obj:
        motion_tensor = motion_obj["x"]
        confidence = motion_obj["confidence"]
        return {
            "x": motion_tensor.float(),
            "confidence": confidence.float(),
        }

    raise ValueError(
        f"Unsupported motion feature format in {motion_path}. Expected a Tensor or a dict with `x` and `confidence`."
    )


def convert_camera_pose_to_9d(frame_motion: torch.Tensor) -> torch.Tensor:
    """Convert per-frame camera poses into 9D features by keeping the 3x3 rotation block."""
    if frame_motion.ndim == 3 and frame_motion.shape[-2:] == (4, 4):
        return frame_motion[:, :3, :3].reshape(frame_motion.shape[0], 9)
    if frame_motion.ndim == 2 and frame_motion.shape[-1] == 16:
        return frame_motion.view(frame_motion.shape[0], 4, 4)[:, :3, :3].reshape(frame_motion.shape[0], 9)
    if frame_motion.ndim == 2 and frame_motion.shape[-1] == 9:
        return frame_motion
    raise ValueError(
        f"Unsupported camera motion shape {tuple(frame_motion.shape)}. Expected [T,4,4], [T,16], or [T,9]."
    )


def aggregate_tracks_to_frames(motion_features: Dict[str, torch.Tensor], is_camera_motion: bool = False) -> torch.Tensor:
    """Pool variable-length track features into frame-level motion features [T, C]."""

    if "frame_motion" in motion_features:
        frame_motion = motion_features["frame_motion"]
        if is_camera_motion:
            return convert_camera_pose_to_9d(frame_motion)
        if frame_motion.ndim > 2:
            frame_motion = frame_motion.reshape(frame_motion.shape[0], -1)
        return frame_motion

    x = motion_features["x"]
    confidence = motion_features["confidence"]
    x_frame = x.squeeze(0).permute(1, 0, 2)  # [T, N_track, C]
    track_conf = confidence.squeeze(0)  # [T, N_track]
    track_weight = torch.softmax(track_conf, dim=-1)
    frame_motion = torch.einsum("tn,tnc->tc", track_weight, x_frame)
    return frame_motion


def sample_video_frames(video_path: str, stride: int = FRAME_STRIDE) -> List[Image.Image]:
    """Sample one frame every `stride` frames from a video file."""
    try:
        import decord

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = list(range(0, total_frames, stride))
        if not frame_indices:
            frame_indices = [0]
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]
    except Exception:
        import torchvision
        from torchvision.transforms import ToPILImage

        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")
        total_frames = int(video.shape[0])
        frame_indices = list(range(0, total_frames, stride))
        if not frame_indices:
            frame_indices = [0]
        return [ToPILImage()(video[idx]) for idx in frame_indices]


def build_motion_token_segments(frame_motion: torch.Tensor, stride: int = FRAME_STRIDE) -> torch.Tensor:
    """Convert frame-level motion [T, C] into one pooled motion vector per stride-sized chunk."""
    num_segments = frame_motion.shape[0] // stride
    if num_segments == 0:
        return frame_motion.new_zeros((0, frame_motion.shape[-1]))
    trimmed_motion = frame_motion[: num_segments * stride]
    return trimmed_motion.view(num_segments, stride, frame_motion.shape[-1]).mean(dim=1)


def build_motion_question(target: str) -> str:
    """Sample a motion-description question template to reduce prompt overfitting."""
    return random.choice(MOTION_PROMPT_TEMPLATES).format(target=target)

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""
    target = example.get("target")
    if target is None:
        raise ValueError("Each sample must provide a `target` field for motion-description SFT.")

    media_path = resolve_media_path(example)
    prepared_example = {
        "video_path": media_path,
        "target": target,
        "solution": example["solution"],
    }

    object_motion_path = example.get("object_motion", example.get("object", example.get("motion_path")))
    camera_motion_path = example.get("camera_motion")
    if object_motion_path is None and camera_motion_path is None:
        raise ValueError(
            "Each sample must provide at least one of `object_motion`, `camera_motion`, legacy `object`, or `motion_path`."
        )

    if object_motion_path is not None:
        prepared_example["object_motion_path"] = object_motion_path
    if camera_motion_path is not None:
        prepared_example["camera_motion_path"] = camera_motion_path

    return prepared_example

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs_batch = []
    object_motion_values_batch = []
    camera_motion_values_batch = []
    has_object_motion = False
    has_camera_motion = False

    for i, example in enumerate(examples):
        try:
            sampled_images = sample_video_frames(example["video_path"])

            object_motion_segments = None
            if example.get("object_motion_path") is not None:
                object_motion_features = load_motion_features(example["object_motion_path"])
                object_frame_motion = aggregate_tracks_to_frames(object_motion_features)
                object_motion_segments = build_motion_token_segments(object_frame_motion)

            camera_motion_segments = None
            if example.get("camera_motion_path") is not None:
                camera_motion_features = load_motion_features(example["camera_motion_path"])
                camera_frame_motion = aggregate_tracks_to_frames(camera_motion_features, is_camera_motion=True)
                camera_motion_segments = build_motion_token_segments(camera_frame_motion)

            segment_lengths = [max(len(sampled_images) - 1, 0)]
            if object_motion_segments is not None:
                segment_lengths.append(object_motion_segments.shape[0])
            if camera_motion_segments is not None:
                segment_lengths.append(camera_motion_segments.shape[0])
            num_motion_tokens = min(segment_lengths)

            sampled_images = sampled_images[: num_motion_tokens + 1] if sampled_images else []
            if object_motion_segments is not None:
                object_motion_segments = object_motion_segments[:num_motion_tokens]
                has_object_motion = True
            if camera_motion_segments is not None:
                camera_motion_segments = camera_motion_segments[:num_motion_tokens]
                has_camera_motion = True

            user_content = []
            for frame_idx, frame_image in enumerate(sampled_images):
                user_content.append({"type": "image", "image": frame_image})
                if frame_idx < num_motion_tokens:
                    user_content.append({"type": "text", "text": MOTION_TOKEN})
            user_content.append(
                {"type": "text", "text": build_motion_question(example["target"])}
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": example["solution"]}]},
            ]

            texts.append(processor.apply_chat_template(messages, tokenize=False))
            image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
            image_inputs_batch.extend(image_inputs if image_inputs is not None else [])
            object_motion_values_batch.append(object_motion_segments)
            camera_motion_values_batch.append(camera_motion_segments)
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs_batch or None,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask all vision placeholder tokens so the loss only applies to text targets.
    visual_tokens = {
        getattr(processor, "image_token_id", None),
        getattr(processor, "video_token_id", None),
        getattr(processor, "vision_start_token_id", None),
        getattr(processor, "vision_end_token_id", None),
    }
    if not visual_tokens or visual_tokens == {None}:
        visual_tokens = {
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        }

    for visual_token_id in visual_tokens:
        if visual_token_id is None:
            continue
        labels[labels == visual_token_id] = -100

    if motion_token_id is not None:
        labels[labels == motion_token_id] = -100

    inputs["labels"] = labels
    if has_object_motion:
        inputs["object_motion_values"] = object_motion_values_batch
        inputs["motion_values"] = object_motion_values_batch
    if has_camera_motion:
        inputs["camera_motion_values"] = camera_motion_values_batch
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
    )
    
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    model.config.motion_feature_dim = MOTION_FEATURE_DIM
    model.config.object_motion_feature_dim = MOTION_FEATURE_DIM
    model.config.camera_motion_feature_dim = 9

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    num_added_tokens = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [MOTION_TOKEN]}
    )
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
    motion_token_id = processor.tokenizer.convert_tokens_to_ids(MOTION_TOKEN)
    model.config.motion_token_id = motion_token_id

    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.motion_projector.parameters():
        param.requires_grad = True
    for param in model.model.camera_motion_projector.parameters():
        param.requires_grad = True
    for param in model.model.motion_fusion.parameters():
        param.requires_grad = True

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params}")

    # Prepare dataset
    prepared_dataset = Dataset.from_list(
        [prepare_dataset(example) for example in dataset["train"]]
    )

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=None,
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model
    if trainer.accelerator.is_main_process:
        num_added_tokens = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [MOTION_TOKEN]}
        )
        if num_added_tokens > 0:
            trainer.model.resize_token_embeddings(len(processor.tokenizer))
        motion_token_id = processor.tokenizer.convert_tokens_to_ids(MOTION_TOKEN)
        trainer.model.config.motion_token_id = motion_token_id

        trainer.save_model(training_args.output_dir)
        processor.tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    if wandb.run is not None:
        wandb.finish()
