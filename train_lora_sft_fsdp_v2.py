#!/usr/bin/env python3
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from train_lora_sft_fsdp import (
    ChatSFTDataset,
    SFTDataCollator,
    infer_lora_target_modules,
    load_jsonl,
    parse_args,
    split_records,
)


def main() -> None:
    args = parse_args()
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

    fsdp = f"{args.fsdp_mode} auto_wrap"
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
        eval_strategy="steps" if eval_dataset else "no",
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
