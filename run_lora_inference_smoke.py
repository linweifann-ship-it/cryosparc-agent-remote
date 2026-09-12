#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for loading Qwen base model with a LoRA adapter.")
    parser.add_argument("--base-model-path", default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def build_demo_messages() -> list[dict[str, str]]:
    model_input = {
        "schema_version": "1.0",
        "task_type": "workflow_decision",
        "dataset": {"dataset_id": "EMPIAR-DEMO"},
        "reference_data": None,
        "workflow": {
            "workflow_id": "wf_demo",
            "workflow_title": "Demo workflow",
            "workflow_version": "1.0.0",
            "current_workflow_node_id": "J5",
            "completed_nodes": [
                {"workflow_node_id": "J1", "job_type": "import_movies", "actual_job_id": "J1", "status": "completed"},
                {"workflow_node_id": "J2", "job_type": "motion_correction_multi", "actual_job_id": "J2", "status": "completed"},
            ],
            "available_upstream_outputs": ["J1.movies", "J2.micrographs"],
        },
        "current_state": {
            "workflow_node_id": "J5",
            "job_type": "patch_ctf_estimation_multi",
            "actual_job_id": "J5",
            "status": "completed",
            "project": {"id": "P0", "title": "Demo project"},
            "job": {"id": "J5", "title": "Patch CTF Estimation", "type": "patch_ctf_estimation_multi", "status": "completed"},
            "inputs": {"micrographs": {"dataset": "J2.micrographs", "fields": {"micrograph_blob": "J2.micrographs.micrograph_blob"}}},
            "parameters": {"General": {"amplitude_contrast": 0.1, "min_res_A": 30, "max_res_A": 5}},
            "outputs": {"micrographs": {"dataset": "J5-G0", "fields": {"ctf": {"type": "exposure.ctf", "passthrough": False}}, "num_items": 1200}},
            "derived_metrics": {"micrographs_num_items": 1200},
            "quality_flags": [],
            "evidence": ["Patch CTF estimation completed successfully."],
            "recent_completed_nodes": ["J5"],
            "reference_data": None,
        },
        "candidate_actions": [
            {
                "action_id": "forward_J6",
                "action_type": "forward",
                "workflow_node_id": "J6",
                "job_type": "blob_picker_gpu",
                "allowed_inputs": ["J5.micrographs"],
                "parameter_template": {"diameter": 180},
            }
        ],
        "constraints": {
            "max_branches": 8,
            "must_choose_from_candidates": True,
            "return_json_only": True,
        },
    }

    return [
        {
            "role": "system",
            "content": (
                "You are a cryoSPARC workflow assistant. Read the workflow state and "
                "return only valid JSON that follows the required decision schema."
            ),
        },
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
    ]


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()

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

    messages = build_demo_messages()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][model_inputs["input_ids"].shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("=== Generation ===")
    print(response_text.strip())


if __name__ == "__main__":
    main()
