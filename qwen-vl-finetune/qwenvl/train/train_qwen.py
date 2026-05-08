# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import shutil
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

NAV_SPECIAL_TOKENS = [
    "<graph>",
    "<node>",
    "</node>",
    "<stop>",
    "<nav>",
    "</nav>",
    "<idx>",
    "</idx>",
]

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def prune_checkpoints(output_dir: str, save_total_limit, best_model_checkpoint=None):
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoint_dirs = []
    for path in pathlib.Path(output_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda item: item[0])
    if len(checkpoint_dirs) <= save_total_limit:
        return

    best_checkpoint = str(Path(best_model_checkpoint)) if best_model_checkpoint else None
    if (
        best_checkpoint is not None
        and save_total_limit == 1
        and str(checkpoint_dirs[-1][1]) != best_checkpoint
    ):
        save_total_limit = 2

    checkpoints_to_delete = checkpoint_dirs[: max(0, len(checkpoint_dirs) - save_total_limit)]
    for _, checkpoint_dir in checkpoints_to_delete:
        logging.info(
            "Deleting older checkpoint [%s] due to save_total_limit=%s",
            checkpoint_dir,
            save_total_limit,
        )
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


class CheckpointPruningTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        prune_checkpoints(
            self._get_output_dir(trial=trial),
            self.args.save_total_limit,
            self.state.best_model_checkpoint,
        )


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    report_to = training_args.report_to
    if not report_to or (isinstance(report_to, str) and report_to.lower() == "none"):
        training_args.report_to = "none"

    local_rank = training_args.local_rank
    if local_rank is None or local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    training_args.local_rank = local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    os.makedirs(training_args.output_dir, exist_ok=True)

    debug_train = os.environ.get("DEBUG_TRAIN") == "1"
    if debug_train:
        print(
            "[Debug] dataset_use=",
            data_args.dataset_use,
            "use_nav_graph=",
            data_args.use_nav_graph,
            "local_rank=",
            local_rank,
        )

    nav_graph_requested = bool(
        data_args.use_nav_graph
        or any(flag in data_args.dataset_use.lower() for flag in ("nav", "dagger"))
    )
    if debug_train:
        print("[Debug] nav_graph_requested=", nav_graph_requested)
    if nav_graph_requested:
        data_args.data_flatten = False
        data_args.data_packing = False
        data_args.use_nav_graph = True
        training_args.attn_implementation = "sdpa"
        print("[Info] Navigation graph mode enabled; forcing sdpa attention and disabling flatten/packing.")

    attn_implementation = attn_implementation or training_args.attn_implementation
    if nav_graph_requested:
        attn_implementation = "sdpa"

    model_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if nav_graph_requested:
        model_config.graph_sprels = True
        if hasattr(model_config, "text_config") and model_config.text_config is not None:
            model_config.text_config.graph_sprels = True
    if debug_train:
        print(
            "[Debug] model_config.graph_sprels=",
            getattr(model_config, "graph_sprels", None),
            "text_config.graph_sprels=",
            getattr(getattr(model_config, "text_config", None), "graph_sprels", None),
        )
    model_config_kwargs = {"config": model_config} if model_config is not None else {}

    model_type = getattr(model_config, "model_type", "").lower()
    model_name = Path(model_args.model_name_or_path.rstrip("/")).name.lower()

    if model_type.startswith("qwen3_vl_moe") or (model_type.startswith("qwen3") and "a" in model_name):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **model_config_kwargs,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif model_type.startswith("qwen3"):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **model_config_kwargs,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif model_type.startswith("qwen2_5"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **model_config_kwargs,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **model_config_kwargs,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    if nav_graph_requested and data_args.model_type not in {"qwen2.5vl", "qwen3vl"}:
        raise NotImplementedError("Navigation graph bias is currently implemented only for Qwen2.5-VL and Qwen3-VL.")

    if (
        training_args.save_total_limit is not None
        and training_args.save_total_limit > 0
        and not training_args.load_best_model_at_end
    ):
        prune_checkpoints(training_args.output_dir, training_args.save_total_limit)

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if (data_args.data_flatten or data_args.data_packing) and not nav_graph_requested:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": NAV_SPECIAL_TOKENS}
    )
    processor.tokenizer = tokenizer
    if added_tokens > 0:
        print(f"[Info] Added {added_tokens} special tokens to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = CheckpointPruningTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    prune_checkpoints(
        training_args.output_dir,
        training_args.save_total_limit,
        trainer.state.best_model_checkpoint,
    )

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
