#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA decision accuracy on an SFT eval set.")
    parser.add_argument("--base-model-path", default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--sft-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--prompt-ablation",
        choices=["none", "no_candidates_no_parameters"],
        default="none",
        help="Optional prompt ablation mode for evaluating reliance on candidate actions / parameters.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--schema-prompt",
        choices=["default", "minimal_v3_strict"],
        default="default",
        help="Optional explicit output-schema reminder injected into the system prompt.",
    )
    return parser.parse_args()


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def normalize_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {
            "job_type": None,
            "parameters": canonicalize(action),
        }
    return {
        "job_type": action.get("job_type"),
        "parameters": canonicalize(action.get("parameters", {})),
    }


def sort_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [normalize_action(action) for action in actions],
        key=lambda item: (
            item.get("job_type") or "",
            json.dumps(item.get("parameters") or {}, ensure_ascii=False, sort_keys=True),
        ),
    )


def extract_core_decision(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_type": obj.get("decision_type"),
        "selected_actions": sort_actions(obj.get("selected_actions", [])),
    }


def extract_action_keys(actions: list[dict[str, Any]]) -> list[str | None]:
    normalized = sort_actions(actions)
    return [item.get("job_type") for item in normalized]


def build_minimal_v3_system_prompt() -> str:
    return (
        "You are a cryoSPARC workflow assistant. "
        "Return only valid JSON. "
        "The output must be exactly one JSON object with these top-level keys only: "
        "schema_version, decision_type, selected_actions. "
        "schema_version must be the string 3.0. "
        "decision_type must be one of: forward, branch, stop. "
        "selected_actions must be a list of objects, and each object must contain only: job_type and parameters. "
        "Do not output any other top-level keys such as decision, next_action, action, reason, explanation, confidence, evidence, rollback_target, branch_plan, or workflow_node_id. "
        "Do not output markdown or prose."
    )


def apply_schema_prompt(messages: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    normalized = [dict(message) for message in messages]
    if mode == "default":
        return normalized
    if mode != "minimal_v3_strict":
        raise ValueError(f"Unsupported schema prompt mode: {mode}")

    strict_system = build_minimal_v3_system_prompt()
    if normalized and normalized[0].get("role") == "system":
        normalized[0]["content"] = strict_system
    else:
        normalized.insert(0, {"role": "system", "content": strict_system})
    return normalized


def ablate_prompt_messages(messages: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode == "none":
        return messages

    if mode != "no_candidates_no_parameters":
        raise ValueError(f"Unsupported prompt ablation mode: {mode}")

    ablated_messages = [dict(message) for message in messages]
    if not ablated_messages:
        return ablated_messages

    user_payload = json.loads(ablated_messages[-1]["content"])
    user_payload["candidate_actions"] = []
    user_payload.setdefault("current_state", {})["parameters"] = {}
    user_payload.setdefault("constraints", {})["must_choose_from_candidates"] = False
    ablated_messages[-1]["content"] = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return ablated_messages


def generate_response(tokenizer: AutoTokenizer, model: PeftModel, prompt_messages: list[dict[str, str]], max_new_tokens: int, disable_thinking: bool) -> str:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not disable_thinking,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][model_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    sft_jsonl = Path(args.sft_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(sft_jsonl, args.limit)

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

    summary = {
        "prompt_ablation": args.prompt_ablation,
        "schema_prompt": args.schema_prompt,
        "sample_count": len(records),
        "valid_json_count": 0,
        "decision_type_match_count": 0,
        "selected_action_count_match_count": 0,
        "selected_action_set_match_count": 0,
        "selected_action_parameters_match_count": 0,
        "core_decision_exact_match_count": 0,
        "full_target_exact_match_count": 0,
        "parse_failures": 0,
    }
    per_sample_path = output_dir / "per_sample_results.jsonl"
    summary_path = output_dir / "summary.json"

    with per_sample_path.open("w", encoding="utf-8") as fout:
        for index, record in enumerate(records):
            sample_id = record["id"]
            messages = record["messages"]
            prompt_messages = apply_schema_prompt(
                ablate_prompt_messages(messages[:-1], args.prompt_ablation),
                args.schema_prompt,
            )
            target_text = messages[-1]["content"]
            target_obj = json.loads(target_text)

            print(f"[{index + 1}/{len(records)}] Evaluating {sample_id}", flush=True)
            response_text = generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=prompt_messages,
                max_new_tokens=args.max_new_tokens,
                disable_thinking=args.disable_thinking,
            )

            parsed_obj = None
            parse_error = None
            try:
                parsed_obj = json.loads(response_text)
                summary["valid_json_count"] += 1
            except Exception as exc:
                parse_error = repr(exc)
                summary["parse_failures"] += 1

            result: dict[str, Any] = {
                "id": sample_id,
                "dataset_id": record.get("dataset_id"),
                "valid_json": parsed_obj is not None,
                "parse_error": parse_error,
                "generated_text": response_text,
            }

            if parsed_obj is not None:
                target_core = extract_core_decision(target_obj)
                pred_core = extract_core_decision(parsed_obj)
                target_action_keys = extract_action_keys(target_obj.get("selected_actions", []))
                pred_action_keys = extract_action_keys(parsed_obj.get("selected_actions", []))

                decision_type_match = parsed_obj.get("decision_type") == target_obj.get("decision_type")
                action_count_match = len(parsed_obj.get("selected_actions", [])) == len(target_obj.get("selected_actions", []))
                action_set_match = pred_action_keys == target_action_keys
                parameters_match = pred_core["selected_actions"] == target_core["selected_actions"]
                core_exact_match = pred_core == target_core
                full_target_exact_match = canonicalize(parsed_obj) == canonicalize(target_obj)

                summary["decision_type_match_count"] += int(decision_type_match)
                summary["selected_action_count_match_count"] += int(action_count_match)
                summary["selected_action_set_match_count"] += int(action_set_match)
                summary["selected_action_parameters_match_count"] += int(parameters_match)
                summary["core_decision_exact_match_count"] += int(core_exact_match)
                summary["full_target_exact_match_count"] += int(full_target_exact_match)

                result.update(
                    {
                        "decision_type_match": decision_type_match,
                        "selected_action_count_match": action_count_match,
                        "selected_action_set_match": action_set_match,
                        "selected_action_parameters_match": parameters_match,
                        "core_decision_exact_match": core_exact_match,
                        "full_target_exact_match": full_target_exact_match,
                        "target_decision_type": target_obj.get("decision_type"),
                        "pred_decision_type": parsed_obj.get("decision_type"),
                        "target_action_keys": target_action_keys,
                        "pred_action_keys": pred_action_keys,
                    }
                )

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    sample_count = summary["sample_count"]
    summary.update(
        {
            "valid_json_rate": summary["valid_json_count"] / sample_count,
            "decision_type_accuracy": summary["decision_type_match_count"] / sample_count,
            "selected_action_count_accuracy": summary["selected_action_count_match_count"] / sample_count,
            "selected_action_set_accuracy": summary["selected_action_set_match_count"] / sample_count,
            "selected_action_parameters_accuracy": summary["selected_action_parameters_match_count"] / sample_count,
            "core_decision_exact_match_accuracy": summary["core_decision_exact_match_count"] / sample_count,
            "full_target_exact_match_accuracy": summary["full_target_exact_match_count"] / sample_count,
        }
    )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== Evaluation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-sample results written to: {per_sample_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
