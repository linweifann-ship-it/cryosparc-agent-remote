#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a LoRA adapter and recover a JSON object from mixed output.")
    parser.add_argument("--base-model-path", default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--sft-jsonl", default=None, help="Optional SFT JSONL file to reuse a real sample prompt.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--input-json", default=None, help="Optional path to a raw model_input JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def load_sample_from_sft(path: Path, sample_index: int) -> tuple[str, list[dict[str, str]]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No samples found in {path}")
    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(f"sample-index {sample_index} out of range for {len(lines)} samples")
    record = json.loads(lines[sample_index])
    messages = record["messages"]
    return record["id"], messages[:-1]


def load_prompt_messages(args: argparse.Namespace) -> tuple[str, list[dict[str, str]]]:
    if args.sft_jsonl:
        return load_sample_from_sft(Path(args.sft_jsonl).resolve(), args.sample_index)

    if args.input_json:
        input_path = Path(args.input_json).resolve()
        model_input = json.loads(input_path.read_text(encoding="utf-8"))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cryoSPARC workflow assistant. "
                    "Return exactly one valid JSON object and no other text."
                ),
            },
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
        ]
        return input_path.stem, messages

    raise ValueError("Provide either --sft-jsonl or --input-json")


def extract_first_json_object(text: str) -> tuple[str | None, str | None]:
    start = text.find("{")
    if start < 0:
        return None, "no_open_brace"

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], None

    return None, "unterminated_json_object"


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    sample_id, prompt_messages = load_prompt_messages(args)

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

    extracted_json_text, extract_error = extract_first_json_object(response_text)
    parsed: Any = None
    parse_error = None
    if extracted_json_text is not None:
        try:
            parsed = json.loads(extracted_json_text)
        except Exception as exc:
            parse_error = repr(exc)

    print(f"=== Sample ID ===\n{sample_id}")
    print("=== Raw Text ===")
    print(response_text)
    print("=== Extracted JSON Text ===")
    print(extracted_json_text if extracted_json_text is not None else "<none>")
    print("=== JSON Recovery Check ===")
    if parsed is not None:
        print("recovered_json: true")
        print(f"top_level_type: {type(parsed).__name__}")
        if isinstance(parsed, dict):
            print("top_level_keys:", sorted(parsed.keys()))
    else:
        print("recovered_json: false")
        print("extract_error:", extract_error)
        print("parse_error:", parse_error)


if __name__ == "__main__":
    main()
