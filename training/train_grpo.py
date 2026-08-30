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

import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from training.trainer import Qwen3VLGRPOTrainerScore
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import numpy as np

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


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def normalize_label(label: str) -> str:
        return str(label).strip().lower().replace(" ", "_").replace("-", "_")

    def canonicalize_regression_key(key: str):
        normalized_key = str(key).strip().lower().replace("-", "_")
        normalized_key_space = normalized_key.replace("_", " ")
        for canonical_key, aliases in REGRESSION_DIMENSION_ALIASES.items():
            if normalized_key in aliases or normalized_key_space in aliases:
                return canonical_key
        return None

    def parse_regression_scores(answer_text: str):
        if not answer_text:
            return {}

        parsed_scores = {}

        try:
            obj = json.loads(answer_text)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    canonical_key = canonicalize_regression_key(key)
                    if canonical_key is None:
                        continue
                    score = normalize_number(str(value))
                    if score is not None:
                        parsed_scores[canonical_key] = score
        except Exception:
            pass

        if parsed_scores:
            return parsed_scores

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

    ISSUE_LABELS = {
        "interpenetration",
        "split_or_merge",
        "pop_in_or_disappear",
        "deformation",
        "texture_flicker",
        "stiff_movement",
        "teleportation",
        "gravity_violation",
        "causality_error",
        "secondary_motion_error",
        "buoyancy_error",
        "energy_conservation_violation",
    }

    def parse_issue_labels(answer_text: str):
        parse_ok = False
        labels = []
        try:
            obj = json.loads(answer_text)
            parse_ok = True
            if isinstance(obj, dict):
                for key in ["issues", "labels", "problems", "artifacts"]:
                    if key in obj:
                        if isinstance(obj[key], list):
                            labels = obj[key]
                        elif isinstance(obj[key], str):
                            labels = [x.strip() for x in obj[key].split(",") if x.strip()]
                        break
            elif isinstance(obj, list):
                labels = obj
            elif isinstance(obj, str):
                labels = [x.strip() for x in obj.split(",") if x.strip()]
        except Exception:
            labels = [x.strip().strip('"').strip("'") for x in re.split(r"[,\n;]+", answer_text) if x.strip()]

        normalized = set()
        invalid = set()
        for x in labels:
            n = normalize_label(x)
            if n in ISSUE_LABELS:
                normalized.add(n)
            else:
                invalid.add(n)
        return normalized, parse_ok, invalid

    def normalize_choice(answer_text: str) -> str:
        text = str(answer_text).strip().upper()
        match = re.match(r"^([A-Z])(?:[\.\)\:\s].*)?$", text)
        if match:
            return match.group(1)
        return text

    question_type_raw = kwargs['problem_type'][0]
    question_type = str(question_type_raw).strip().lower()
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                pred = normalize_choice(output_ans)
                gt = normalize_choice(gt_ans)
                reward = 1.0 if pred == gt and len(pred) == 1 and pred.isalpha() else 0.0
            elif question_type == "discrimination":
                pred = output_ans.strip().lower()
                gt = gt_ans.strip().lower()
                alias = {"true": "real", "false": "fake", "authentic": "real", "synthetic": "fake", "generated": "fake"}
                pred = alias.get(pred, pred)
                gt = alias.get(gt, gt)
                reward = 1.0 if pred == gt and pred in {"real", "fake"} else 0.0
            elif question_type == "regression":
                # 可调参数
                SCORE_MIN = 1.0        # 数据集分数下界
                SCORE_MAX = 5.0        # 数据集分数上界
                TAU_ABS = 0.2          # 绝对误差阈值（< 0.2 视为“命中”）
                W_CONT = 0.8           # 连续奖励权重
                W_BIN  = 0.2           # 二值（阈值）奖励权重
                PENALTY = 1.5          # 高估时的额外惩罚

                target_keys = [
                    "structural_stability",
                    "physical_plausibility",
                    "motion_coherence",
                ]
                gt_scores = parse_regression_scores(gt_ans)
                pred_scores = parse_regression_scores(output_ans)

                if not all(key in gt_scores for key in target_keys) or not all(key in pred_scores for key in target_keys):
                    reward = 0.0
                else:
                    dimension_rewards = []
                    score_span = max(1e-6, SCORE_MAX - SCORE_MIN)

                    for key in target_keys:
                        gt_number = gt_scores[key]
                        out_number = pred_scores[key]

                        if not np.isfinite(gt_number) or not np.isfinite(out_number):
                            dimension_rewards = []
                            break

                        diff = out_number - gt_number
                        abs_diff = float(abs(diff))
                        penalty = abs_diff * (PENALTY if diff > 0 else 1.0)

                        # 对 1~5 的评分区间做归一化，避免低分样本因 gt 小而被过度惩罚。
                        norm_abs_diff = float(np.clip(penalty / score_span, 0.0, 1.0))
                        r_cont = 1.0 - norm_abs_diff ** 2      # ∈[0,1]

                        # 二值阈值奖励：更“有难度”的通过奖
                        r_bin = 1.0 if abs_diff < TAU_ABS else 0.0

                        # 组合：连续奖励保留梯度，阈值奖励鼓励精确命中。
                        dimension_rewards.append(W_CONT * r_cont + W_BIN * r_bin)

                    reward = float(dimension_rewards[0]*dimension_rewards[1]*dimension_rewards[2])
            
            elif question_type == "cause-inspection":
                pred_set, parse_ok, invalid_pred = parse_issue_labels(output_ans)
                gt_set, _, _ = parse_issue_labels(gt_ans)

                if len(pred_set) == 0 and len(gt_set) == 0:
                    reward = 1.0
                else:
                    tp = len(pred_set & gt_set)
                    precision = tp / max(len(pred_set), 1)
                    recall = tp / max(len(gt_set), 1)
                    reward = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    # pattern = (
    # r"(?s)^"                     # DOTALL: 让 . 能匹配换行
    # r"<thinking>\s*\n?"             # <thinking> 后可有可无换行
    # r".*?"                       # think 内容（非贪婪、可跨行）
    # r"\s*</thinking>\s*\n?"         # </thinking> 后可有可无换行
    # r"<answer>\s*\n?"            # <answer> 后可有可无换行
    # r"\s*</answer>\s*$"          # 直到 </answer> 结束
    # )

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., "
    "<thinking> reasoning process here </thinking><answer> answer here </answer>. "
    # "The input video contains visualized tracks overlaid on the frames, which should be used as part of the analysis."
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    QUESTION_TEMPLATE = (
        "{Question}\n"
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide the thinking process within the <thinking> </thinking> tags and answer with only the correct option letter (such as A, B, C, or D) within the <answer> </answer> tags.",
        "regression": " Please provide the thinking process within the <thinking> </thinking> tags and output only the three scores within the <answer> </answer> tags, one decimal place each, using this exact format: {\"structural_stability\": 3.2, \"physical_plausibility\": 3.5, \"motion_coherence\": 2.8}. Be strict and conservative in scoring. A mostly static object is not necessarily poor.",
        "discrimination": "Please provide the thinking process within the <thinking> </thinking> tags and answer only Real or Fake within the <answer> </answer> tags.",
        "cause-inspection": "Please provide the thinking process within the <thinking> </thinking> tags and provide a strict JSON object in <answer> </answer> as {\"issues\": [\"interpenetration\", \"split_or_merge\"]}. Allowed labels are: interpenetration, split_or_merge, pop_in_or_disappear, deformation, texture_flicker, stiff_movement, teleportation, gravity_violation, causality_error, secondary_motion_error, buoyancy_error, energy_conservation_violation. Use {\"issues\": []} when there is no issue.",
        }

    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                        },
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT + QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    if training_args.use_vllm:
        raise ValueError("The released motion-aware trainer does not support --use_vllm.")
    trainer_cls = Qwen3VLGRPOTrainerScore
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
