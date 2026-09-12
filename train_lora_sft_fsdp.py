#!/usr/bin/env python3
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSDP + LoRA SFT entrypoint for cryoagent workflow decision data.")
    parser.add_argument("--model-path", default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test")
    parser.add_argument(
        "--init-adapter-path",
        default=None,
        help="Optional adapter checkpoint to merge into the base model before starting SFT.",
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to workflow_sft_data.jsonl")
    parser.add_argument("--eval-jsonl", default=None, help="Optional separate eval JSONL file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument(
        "--fsdp-mode",
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "hybrid_shard_zero2"],
        help="Trainer FSDP sharding mode.",
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen3_5DecoderLayer",
        help="Transformer decoder layer class name for FSDP auto wrapping.",
    )
    parser.add_argument("--fsdp-offload-params", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_records(records: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record.get("workflow_id") or record.get("dataset_id") or record.get("id")
        grouped.setdefault(key, []).append(record)

    keys = list(grouped.keys())
    random.Random(seed).shuffle(keys)
    eval_group_count = max(1, math.ceil(len(keys) * eval_ratio)) if len(keys) > 1 else 0
    eval_keys = set(keys[:eval_group_count])

    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        if key in eval_keys:
            eval_records.extend(bucket)
        else:
            train_records.extend(bucket)
    if not train_records:
        raise ValueError("Train split is empty after grouping.")
    return train_records, eval_records


def apply_chat_template(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


class ChatSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        for record in records:
            messages = record["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                skipped += 1
                continue
            prompt_text = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
            full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            if len(prompt_ids) >= len(full_ids):
                skipped += 1
                continue

            labels = [-100] * len(full_ids)
            labels[len(prompt_ids):] = full_ids[len(prompt_ids):]
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
        if not self.examples:
            raise ValueError("No usable SFT samples were produced after tokenization.")
        print(f"Prepared {len(self.examples)} samples; skipped {skipped}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


@dataclass
class SFTDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def infer_lora_target_modules(model, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "out_proj",
        "gate_up_proj",
    }
    discovered = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in preferred:
                discovered.add(suffix)
    if not discovered:
        raise ValueError("Could not infer LoRA target modules automatically.")
    return sorted(discovered)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path).resolve()
    train_jsonl = Path(args.train_jsonl).resolve()
    eval_jsonl = Path(args.eval_jsonl).resolve() if args.eval_jsonl else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_jsonl(train_jsonl)
    if eval_jsonl:
        eval_records = load_jsonl(eval_jsonl)
    else:
        train_records, eval_records = split_records(train_records, args.eval_ratio, args.seed)

    train_dataset = ChatSFTDataset(train_records, tokenizer, args.max_length)
    eval_dataset = ChatSFTDataset(eval_records, tokenizer, args.max_length) if eval_records else None

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = infer_lora_target_modules(model, args.target_modules)
    print(f"Using LoRA target modules: {target_modules}")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.print_trainable_parameters()

    fsdp = [args.fsdp_mode, "auto_wrap"]
    fsdp_config = {
        "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "limit_all_gathers": True,
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
        "use_orig_params": False,
        "offload_params": args.fsdp_offload_params,
    }
    print(f"Enabled FSDP: mode={args.fsdp_mode}, wrap={args.fsdp_transformer_layer_cls_to_wrap}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=args.report_to if args.report_to != "none" else [],
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
