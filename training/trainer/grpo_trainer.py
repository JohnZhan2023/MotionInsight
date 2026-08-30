# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
import random

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from PIL import Image
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration, 
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
MOTION_TOKEN = "<|motion|>"
FRAME_STRIDE = 16


def load_motion_features(motion_path: str) -> Dict[str, torch.Tensor]:
    try:
        motion_obj = torch.load(motion_path, map_location="cpu", weights_only=True)
    except TypeError:
        motion_obj = torch.load(motion_path, map_location="cpu")

    if isinstance(motion_obj, torch.Tensor):
        return {"frame_motion": motion_obj.float()}

    if "x" in motion_obj and "confidence" in motion_obj:
        return {
            "x": motion_obj["x"].float(),
            "confidence": motion_obj["confidence"].float(),
        }

    raise ValueError(
        f"Unsupported motion feature format in {motion_path}. Expected a Tensor or a dict with `x` and `confidence`."
    )


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


def aggregate_tracks_to_frames(motion_features: Dict[str, torch.Tensor], is_camera_motion: bool = False) -> torch.Tensor:
    if "frame_motion" in motion_features:
        frame_motion = motion_features["frame_motion"]
        if is_camera_motion:
            return convert_camera_pose_to_9d(frame_motion)
        if frame_motion.ndim > 2:
            frame_motion = frame_motion.reshape(frame_motion.shape[0], -1)
        return frame_motion

    x = motion_features["x"]
    confidence = motion_features["confidence"]
    x_frame = x.squeeze(0).permute(1, 0, 2)
    track_conf = confidence.squeeze(0)
    track_weight = torch.softmax(track_conf, dim=-1)
    return torch.einsum("tn,tnc->tc", track_weight, x_frame)


def build_motion_token_segments(frame_motion: torch.Tensor, stride: int = FRAME_STRIDE) -> torch.Tensor:
    num_segments = frame_motion.shape[0] // stride
    if num_segments == 0:
        return frame_motion.new_zeros((0, frame_motion.shape[-1]))
    trimmed_motion = frame_motion[: num_segments * stride]
    return trimmed_motion.view(num_segments, stride, frame_motion.shape[-1]).mean(dim=1)


def sample_video_frames(video_path: str, stride: int = FRAME_STRIDE) -> List[Image.Image]:
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


class Qwen3VLGRPOTrainerScore(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            # model_init_kwargs["use_cache"] = (
            #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            # )
            if "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #self.ref_model = None
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        if hasattr(processing_class, "tokenizer"):
            num_added_tokens = processing_class.tokenizer.add_special_tokens(
                {"additional_special_tokens": [MOTION_TOKEN]}
            )
            if num_added_tokens > 0:
                model.resize_token_embeddings(len(processing_class.tokenizer))
            motion_token_id = processing_class.tokenizer.convert_tokens_to_ids(MOTION_TOKEN)
            model.config.motion_token_id = motion_token_id
            if self.ref_model is not None:
                self.ref_model.config.motion_token_id = motion_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal
        self.epsilon_low  = 0.2
        self.epsilon_high = 0.2

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _repeat_sample_list(self, values, repeat_count):
        if values is None:
            return None
        repeated = []
        for value in values:
            repeated.extend([value] * repeat_count)
        return repeated

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    

        batch_data_type = inputs[0].get("data_type", "video")
        batch_targets = [example.get("target") for example in inputs]
        batch_object_motions = [example.get("object_motion") for example in inputs]
        batch_camera_motions = [example.get("camera_motion") for example in inputs]
        object_motion_values_batch = []
        camera_motion_values_batch = []
        for object_motion_path, camera_motion_path in zip(batch_object_motions, batch_camera_motions):
            object_motion_segments = None
            if object_motion_path is not None:
                object_motion_features = load_motion_features(object_motion_path)
                object_frame_motion = aggregate_tracks_to_frames(object_motion_features)
                object_motion_segments = build_motion_token_segments(object_frame_motion)
            object_motion_values_batch.append(object_motion_segments)

            camera_motion_segments = None
            if camera_motion_path is not None:
                camera_motion_features = load_motion_features(camera_motion_path)
                camera_frame_motion = aggregate_tracks_to_frames(camera_motion_features, is_camera_motion=True)
                camera_motion_segments = build_motion_token_segments(camera_frame_motion)
            camera_motion_values_batch.append(camera_motion_segments)

        prompts = []
        prompt_examples = []
        prompts_text = []
        image_inputs_batch = []
        video_inputs_batch = []
        video_kwargs_list = []

        for idx, example in enumerate(inputs):
            media_path = example.get("video_path", example.get("path"))
            prompt_copy = copy.deepcopy(example["prompt"])
            prompt_copy = self.remove_none_from_data(prompt_copy)

            if batch_data_type == "video":
                sampled_images = sample_video_frames(media_path)
                object_motion = object_motion_values_batch[idx]
                camera_motion = camera_motion_values_batch[idx]
                segment_lengths = [max(len(sampled_images) - 1, 0)]
                if object_motion is not None:
                    segment_lengths.append(object_motion.shape[0])
                if camera_motion is not None:
                    segment_lengths.append(camera_motion.shape[0])
                num_motion_tokens = min(segment_lengths)

                sampled_images = sampled_images[: num_motion_tokens + 1] if sampled_images else []
                if object_motion is not None:
                    object_motion_values_batch[idx] = object_motion[:num_motion_tokens]
                if camera_motion is not None:
                    camera_motion_values_batch[idx] = camera_motion[:num_motion_tokens]

                question_text = prompt_copy[0]["content"][-1]["text"]
                user_content = []
                for frame_idx, frame_image in enumerate(sampled_images):
                    user_content.append({"type": "image", "image": frame_image})
                    if frame_idx < num_motion_tokens:
                        user_content.append({"type": "text", "text": MOTION_TOKEN})
                user_content.append({"type": "text", "text": question_text})
                prompt_copy[0]["content"] = user_content
            elif batch_data_type == "image":
                prompt_copy[0]["content"][0]["image"] = media_path

            prompt_example = copy.deepcopy(example)
            prompt_example["prompt"] = prompt_copy
            prompt_examples.append(prompt_example)
            prompts.append(prompt_copy)
            prompts_text.append(maybe_apply_chat_template(prompt_example, self.processing_class)["prompt"])

            example_image_inputs, example_video_inputs, example_video_kwargs = process_vision_info(
                prompt_copy, return_video_kwargs=True
            )
            if example_image_inputs is not None:
                image_inputs_batch.extend(example_image_inputs)
            if example_video_inputs is not None:
                video_inputs_batch.extend(example_video_inputs)
            video_kwargs_list.append(example_video_kwargs)

        image_inputs = image_inputs_batch or None
        video_inputs = video_inputs_batch or None
        video_kwargs = video_kwargs_list

        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        if any(value is not None for value in object_motion_values_batch):
            prompt_inputs["object_motion_values"] = object_motion_values_batch
            prompt_inputs["motion_values"] = object_motion_values_batch
        if any(value is not None for value in camera_motion_values_batch):
            prompt_inputs["camera_motion_values"] = camera_motion_values_batch
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            
        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            if any(value is not None for value in object_motion_values_batch):
                shuffled_prompt_inputs["object_motion_values"] = object_motion_values_batch
                shuffled_prompt_inputs["motion_values"] = object_motion_values_batch
            if any(value is not None for value in camera_motion_values_batch):
                shuffled_prompt_inputs["camera_motion_values"] = camera_motion_values_batch
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generation_model = unwrapped_model.get_base_model() if hasattr(unwrapped_model, "get_base_model") else unwrapped_model
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generate_start = time.perf_counter()
            prompt_completion_ids = generation_model.generate(**prompt_inputs, generation_config=self.generation_config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generate_elapsed = time.perf_counter() - generate_start
            if os.getenv("LOG_INFERENCE_TIME") == "true":
                log_path = os.getenv("LOG_PATH")
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"------------- VLM inference time: {generate_elapsed:.6f}s "
                            f"rank={self.accelerator.process_index} "
                            f"batch_size={len(prompts)} "
                            f"num_generations={self.num_generations} "
                            f"outputs={len(prompt_completion_ids)} -------------\n"
                        )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            repeated_targets = self._repeat_sample_list(batch_targets, self.num_generations)
            repeated_object_motions = self._repeat_sample_list(batch_object_motions, self.num_generations)
            repeated_camera_motions = self._repeat_sample_list(batch_camera_motions, self.num_generations)
            repeated_object_motion_values = self._repeat_sample_list(object_motion_values_batch, self.num_generations)
            repeated_camera_motion_values = self._repeat_sample_list(camera_motion_values_batch, self.num_generations)
            
            if self.temporal:
                
                if video_inputs:
            
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    shuffled_generate_start = time.perf_counter()
                    shuffled_prompt_completion_ids = generation_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    shuffled_generate_elapsed = time.perf_counter() - shuffled_generate_start
                    if os.getenv("LOG_INFERENCE_TIME") == "true":
                        log_path = os.getenv("LOG_PATH")
                        if log_path:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(
                                    f"------------- VLM shuffled inference time: {shuffled_generate_elapsed:.6f}s "
                                    f"rank={self.accelerator.process_index} "
                                    f"batch_size={len(prompts)} "
                                    f"num_generations={self.shuffled_num_generations} "
                                    f"outputs={len(shuffled_prompt_completion_ids)} -------------\n"
                                )
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                    shuffled_targets = self._repeat_sample_list(batch_targets, self.shuffled_num_generations)
                    shuffled_object_motions = self._repeat_sample_list(batch_object_motions, self.shuffled_num_generations)
                    shuffled_camera_motions = self._repeat_sample_list(batch_camera_motions, self.shuffled_num_generations)
                    shuffled_object_motion_values = self._repeat_sample_list(object_motion_values_batch, self.shuffled_num_generations)
                    shuffled_camera_motion_values = self._repeat_sample_list(camera_motion_values_batch, self.shuffled_num_generations)
                    
                else:
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dummy_generate_start = time.perf_counter()
                    shuffled_prompt_completion_ids = generation_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dummy_generate_elapsed = time.perf_counter() - dummy_generate_start
                    if os.getenv("LOG_INFERENCE_TIME") == "true":
                        log_path = os.getenv("LOG_PATH")
                        if log_path:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(
                                    f"------------- VLM dummy inference time: {dummy_generate_elapsed:.6f}s "
                                    f"rank={self.accelerator.process_index} "
                                    f"batch_size={len(prompts)} "
                                    f"num_generations=1 "
                                    f"outputs={len(shuffled_prompt_completion_ids)} -------------\n"
                                )
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if "pixel_values" in prompt_inputs:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        

        if "pixel_values_videos" in prompt_inputs:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)
        if any(value is not None for value in repeated_object_motion_values):
            prompt_inputs["object_motion_values"] = repeated_object_motion_values
            prompt_inputs["motion_values"] = repeated_object_motion_values
        if any(value is not None for value in repeated_camera_motion_values):
            prompt_inputs["camera_motion_values"] = repeated_camera_motion_values
        
        
        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                # ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if os.getenv("DEBUG_MODE") == "true" and completions:
            print("sample completion:", completions[0])
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        

        
        
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.2
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)
    
        
        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    ## if 320 <= lenth_list[idx] <= 512:
                    if 128 <= lenth_list[idx] <= 256:
                        rewards[idx] += 0.2

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # if self.len_control and len(selected_indices) == self.num_generations:
        #     for idx in range(len(rewards)):
        #         advantages[idx] += (mem_rewards[idx] - 0.2) * 2

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
       
            
        # import pdb
        # pdb.set_trace()

        # Log the metrics
        # `gather_for_metrics` trims duplicated samples from the final distributed batch.
        # Here tensors are already expanded by `num_generations`, so trimming can break the
        # grouping structure (e.g. keep only 2 rows out of a logical group of 8).
        completion_length = self.accelerator.gather(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather(temporal_rewards_list).mean().item())
        
        self._metrics["reward"].append(self.accelerator.gather(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        

        return loss

    # Use for GSPO
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("The GRPOTrainer does not support returning outputs")

    #     # ==== 0) 准备输入（沿用你现有逻辑，不改） ====
    #     prompts = [x["prompt"] for x in inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

    #     import copy
    #     input_copy = copy.deepcopy(inputs[0]['prompt'])
    #     input_copy = self.remove_none_from_data(input_copy)

    #     if inputs[0]['data_type'] == 'image':
    #         input_copy[0]['content'][0]['image'] = inputs[0]['path']
    #     elif inputs[0]['data_type'] == 'video':
    #         input_copy[0]['content'][0]['video'] = inputs[0]['path']

    #     image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)

    #     prompt_inputs = self.processing_class(
    #         text=copy.deepcopy(prompts_text),
    #         images=image_inputs,
    #         videos=video_inputs,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         add_special_tokens=False,
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)

    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # 如果启用 temporal，对视频 token 做打乱生成“打乱版”完成（仅用于奖励对比），沿用你现有逻辑
    #     if self.temporal and video_inputs:
    #         import torch
    #         indices = torch.randperm(video_inputs[0].size(0))
    #         shuffled_video_inputs = [video_inputs[0][indices]]
    #         shuffled_prompt_inputs = self.processing_class(
    #             text=copy.deepcopy(prompts_text),
    #             images=image_inputs,
    #             videos=shuffled_video_inputs,
    #             return_tensors="pt",
    #             padding=True,
    #             padding_side="left",
    #             add_special_tokens=False,
    #         )
    #         shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
    #         shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
    #         if self.max_prompt_length is not None:
    #             shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
    #             shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]

    #     # ==== 1) 生成完成 ====
    #     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    #         prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]
    #         prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

    #         if self.temporal:
    #             if video_inputs:
    #                 shuffled_prompt_completion_ids = unwrapped_model.generate(
    #                     **shuffled_prompt_inputs, generation_config=self.shuffled_generation_config
    #                 )
    #                 shuffled_prompt_length = shuffled_prompt_ids.size(1)
    #                 shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
    #                 shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
    #                 shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
    #             else:
    #                 shuffled_prompt_completion_ids = unwrapped_model.generate(
    #                     **prompt_inputs, generation_config=self.dummy_generation_config
    #                 )

    #     # ==== 2) 构造 completion_mask（沿用不变） ====
    #     import torch
    #     device = self.accelerator.device
    #     is_eos = completion_ids == self.processing_class.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     has_eos = is_eos.any(dim=1)
    #     eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()  # (B*G, T)

    #     # ==== 3) 整理生成时的多模态张量重复（沿用不变） ====
    #     prompt_inputs.pop("input_ids", None)
    #     prompt_inputs.pop("attention_mask", None)

    #     if inputs[0]['data_type'] == 'image':
    #         prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
    #         prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)

    #     if inputs[0]['data_type'] == 'video':
    #         prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
    #         prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
    #         if 'second_per_grid_ts' in prompt_inputs:
    #             # 该键是非张量/列表，去掉避免 broadcast 问题
    #             del prompt_inputs["second_per_grid_ts"]

    #     # ==== 4) GSPO：序列级 logp / 参考 logp  ====
    #     # 4.1 当前模型每 token 的 logp
    #     try:
    #         per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)  # (B*G, P+C-1)
    #         per_token_logps = per_token_logps[:, prompt_length - 1 :]  # 只保留 completion 段
    #     except Exception as e:
    #         print(f"Error computing per_token_logps: {e}. Fallback without vision tensors.")
    #         per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
    #         per_token_logps = per_token_logps[:, prompt_length - 1 :]

    #     # 4.2 参考模型（old policy）每 token 的 logp
    #     with torch.inference_mode():
    #         try:
    #             if self.ref_model is not None:
    #                 ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
    #             else:
    #                 with self.accelerator.unwrap_model(model).disable_adapter():
    #                     ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
    #             ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
    #         except Exception as e:
    #             print(f"Error computing ref_per_token_logps: {e}. Fallback without vision tensors.")
    #             with self.accelerator.unwrap_model(model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
    #             ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

    #     # ==== 5) 奖励（沿用你当前的奖励流程与 temporal 逻辑） ====
    #     completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     print("zxy3", completions[0])
    #     if is_conversational(inputs[0]):
    #         completions = [[{"role": "assistant", "content": completion}] for completion in completions]

    #     prompts_rep = [p for p in prompts for _ in range(self.num_generations)]
    #     rewards_per_func = torch.zeros(len(prompts_rep), len(self.reward_funcs), device=device)

    #     for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
    #         reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
    #         for key in reward_kwargs:
    #             for example in inputs:
    #                 reward_kwargs[key].extend([example[key]] * self.num_generations)
    #         out = reward_func(prompts=prompts_rep, completions=completions, **reward_kwargs)
    #         rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

    #     if self.temporal and video_inputs:
    #         # 生成打乱版，用于 temporal 对比（复用你原有逻辑）
    #         shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
    #         if is_conversational(inputs[0]):
    #             shuffled_completions = [[{"role": "assistant", "content": s}] for s in shuffled_completions]
    #         shuffled_prompts_rep = [p for p in prompts for _ in range(self.shuffled_num_generations)]
    #         shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts_rep), len(self.reward_funcs), device=device)
    #         for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
    #             sh_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
    #             for key in sh_kwargs:
    #                 for example in inputs:
    #                     sh_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
    #             sh_out = reward_func(prompts=shuffled_prompts_rep, completions=shuffled_completions, **sh_kwargs)
    #             shuffled_rewards_per_func[:, i] = torch.tensor(sh_out, dtype=torch.float32, device=device)

    #         # 你的 temporal 奖励门控逻辑
    #         temporal_rewards_per_func = rewards_per_func.clone()
    #         acc_mean = temporal_rewards_per_func[:, 0].mean()
    #         shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()
    #         if acc_mean >= 0.8 * shuffled_acc_mean:
    #             mask_boost = temporal_rewards_per_func[:, 0] > 0.1
    #             temporal_rewards_per_func[mask_boost, 0] = temporal_rewards_per_func[mask_boost, 0] + 0.2
    #             temporal_rewards = torch.tensor([1.0], device=device)
    #         else:
    #             temporal_rewards = torch.tensor([0.0], device=device)
    #     else:
    #         temporal_rewards = torch.tensor([0.5], device=device)

    #     # 汇总 reward（是否用 temporal 加权沿用不变）
    #     if self.temporal and video_inputs:
    #         rewards = temporal_rewards_per_func.sum(dim=1)
    #     else:
    #         rewards = rewards_per_func.sum(dim=1)

    #     # 可选长度控制（沿用不变）
    #     if self.len_control:
    #         mask_sel = rewards_per_func[:, 0] > 0.1
    #         lenth_list = completion_mask.sum(1)
    #         selected = torch.nonzero(mask_sel, as_tuple=True)[0].tolist()
    #         if len(selected) > 1:
    #             for idx in selected:
    #                 if 320 <= lenth_list[idx] <= 512:
    #                     rewards[idx] += 0.2

    #     # 组内标准化优势（沿用你原来的做法）
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)  # (B*G,)

    #     # ==== 6) GSPO 核心：序列级 PPO 目标 ====
    #     # 6.1 计算序列对数似然（对 completion 段的 token logp 按 mask 求和）
    #     seq_logp = (per_token_logps * completion_mask).sum(dim=1)  # (B*G,)
    #     ref_seq_logp = (ref_per_token_logps * completion_mask).sum(dim=1)  # (B*G,)

    #     # importance sampling 比率（序列级）
    #     log_ratio_seq = seq_logp - ref_seq_logp              # log r
    #     ratio_seq = torch.exp(log_ratio_seq)                 # r = exp(log r)

    #     # 双边裁剪（PPO）
    #     # epsilon_low / epsilon_high 与 TRL 一致：允许不对称裁剪
    #     coef_1 = ratio_seq
    #     coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

    #     # 两个候选项
    #     loss1 = coef_1 * advantages
    #     loss2 = coef_2 * advantages

    #     # GSPO：取 min 后取负号（想最大化优势）
    #     seq_loss = -torch.min(loss1, loss2)  # (B*G,)

    #     # 可选 KL 正则（与原实现一致；这里用 token 级 KL 累加成序列平均/或直接用 token 平均）
    #     if getattr(self, "beta", 0.0) != 0.0:
    #         # token 级 KL：exp(ref - cur) - (ref - cur) - 1，然后按 mask 求平均（与原始保持一致）
    #         x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
    #         per_token_kl = torch.exp(x_clamped) - x_clamped - 1
    #         mean_kl_per_seq = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
    #         seq_loss = seq_loss + self.beta * mean_kl_per_seq

    #     # 归一化（与原先各类 loss_type 的梯度缩放行为对齐。这里采用最常见的 GRPO/BNPO 写法之一）
    #     if getattr(self, "loss_type", "grpo") == "grpo":
    #         # 按每条序列的有效 token 数做平均，再对 batch 取均值
    #         normalizer = completion_mask.sum(dim=1).clamp(min=1.0)
    #         loss = (seq_loss / normalizer).mean()
    #         loss = loss / self.current_gradient_accumulation_steps
    #     elif self.loss_type == "bnpo":
    #         loss = seq_loss.sum() / completion_mask.sum().clamp(min=1.0)
    #         loss = loss / self.current_gradient_accumulation_steps
    #     elif self.loss_type == "dr_grpo":
    #         loss = seq_loss.sum() / (seq_loss.size(0) * self.max_completion_length)
    #         loss = loss / self.current_gradient_accumulation_steps
    #     elif self.loss_type == "dapo":
    #         normalizer = (num_items_in_batch if num_items_in_batch is not None else seq_loss.size(0))
    #         normalizer = normalizer / self.accelerator.num_processes
    #         loss = seq_loss.sum() / normalizer
    #     else:
    #         # 默认退回 grpo 归一化
    #         normalizer = completion_mask.sum(dim=1).clamp(min=1.0)
    #         loss = (seq_loss / normalizer).mean()
    #         loss = loss / self.current_gradient_accumulation_steps

    #     # ==== 7) 记录指标（尽量复用你原先的 log 口径） ====
    #     self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())

    #     # 记录 reward 相关
    #     reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, PreTrainedModel):
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     gathered_rewards = self.accelerator.gather_for_metrics(rewards)
    #     num_devices = gathered_rewards.size(0) // self.num_generations
    #     rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
    #     wrong_devices = (rewards_per_device <= 1).all(dim=1)
    #     wrong_ratio = wrong_devices.sum().item() / num_devices
    #     correct_devices = (rewards_per_device >= 2).all(dim=1)
    #     correct_ratio = correct_devices.sum().item() / num_devices
    #     self._metrics["all_wrong"].append(wrong_ratio)
    #     self._metrics["all_correct"].append(correct_ratio)

    #     if self.temporal:
    #         temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
    #         self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())

    #     self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
    #     self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

    #     if getattr(self, "beta", 0.0) != 0.0:
    #         mean_kl = (( (torch.exp(torch.clamp(ref_per_token_logps - per_token_logps, -10, 10))
    #                     - (ref_per_token_logps - per_token_logps) - 1) * completion_mask ).sum(dim=1)
    #                 / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
    #         self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    #     return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
