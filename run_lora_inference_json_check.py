#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict-JSON inference check with a trained LoRA adapter.")
    parser.add_argument("--base-model-path", default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--sft-jsonl", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def load_sample_messages(path: Path, sample_index: int) -> tuple[str, list[dict[str, str]], str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No samples found in {path}")
    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(f"sample-index {sample_index} out of range for {len(lines)} samples")

    record = json.loads(lines[sample_index])
    messages = record["messages"]
    if len(messages) < 3:
        raise ValueError("Expected at least system/user/assistant messages in SFT record")
    return record["id"], messages[:-1], messages[-1]["content"]


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    sft_jsonl = Path(args.sft_jsonl).resolve()

    sample_id, prompt_messages, expected_output = load_sample_messages(sft_jsonl, args.sample_index)

    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][model_inputs["input_ids"].shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"=== Sample ID ===\n{sample_id}")
    print("=== Generated Text ===")
    print(response_text)

    parsed = None
    error = None
    try:
        parsed = json.loads(response_text)
    except Exception as exc:
        error = repr(exc)

    print("=== JSON Check ===")
    if parsed is not None:
        print("valid_json: true")
        print(f"top_level_type: {type(parsed).__name__}")
        if isinstance(parsed, dict):
            print("top_level_keys:", sorted(parsed.keys()))
    else:
        print("valid_json: false")
        print("parse_error:", error)

    print("=== Reference Target Prefix ===")
    print(expected_output[:600])


if __name__ == "__main__":
    main()
