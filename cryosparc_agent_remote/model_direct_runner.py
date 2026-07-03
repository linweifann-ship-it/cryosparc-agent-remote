# Runs the local Qwen LoRA model and converts its text output into V2 decisions.
import json
import re
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a CryoSPARC workflow decision model. Return exactly one valid JSON "
    "object. Do not include markdown, comments, explanations, or thinking text."
)


def build_workflow_decision_prompt(model_input: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for the direct model closed-loop test."""
    output_contract = {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop",
        "action": "CryoSPARC job type for forward/branch decisions, or omit for stop.",
        "parameters": "Only parameters that should override defaults.",
        "reason": "Short reason for the decision.",
        "confidence": "Number from 0.0 to 1.0.",
        "risk_flags": [],
        "evidence": [],
    }
    user_content = {
        "instruction": (
            "Choose the next workflow action from the current CryoSPARC state. "
            "If the next step is unclear, return a stop decision."
        ),
        "model_input": model_input,
        "output_contract": output_contract,
        "valid_examples": [
            {
                "schema_version": "2.0",
                "decision_type": "forward",
                "action": "class_2D_new",
                "parameters": {"compute_num_gpus": 4, "class2D_K": 50},
                "reason": "Particles are extracted and ready for 2D classification.",
                "confidence": 0.85,
                "risk_flags": [],
                "evidence": ["The last job completed with a particles output."],
            },
            {
                "schema_version": "2.0",
                "decision_type": "stop",
                "reason": "No safe next action is clear from the current state.",
                "confidence": 0.5,
                "risk_flags": ["needs_human_review"],
                "evidence": [],
            },
        ],
    }
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render Qwen chat messages while disabling thinking when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_qwen_lora_model(
    messages: list[dict[str, str]],
    base_model_path: str,
    adapter_path: str | None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Load the base model plus optional LoRA adapter and generate raw text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = normalize_optional_path(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompt = render_chat_prompt(tokenizer, messages)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "raw_text": raw_text,
    }


def normalize_optional_path(path: str | None) -> str | None:
    """Treat empty strings and common sentinel values as no adapter."""
    if path is None:
        return None
    stripped = path.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def parse_model_decision_text(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from model text."""
    json_text = extract_first_json_object(clean_model_text(text))
    return json.loads(json_text)


def clean_model_text(text: str) -> str:
    """Remove common non-JSON wrappers from model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^/think\b.*?(?=\{)", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    return cleaned


def extract_first_json_object(text: str) -> str:
    """Return the first balanced JSON object from a larger string."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No balanced JSON object found in model output.")
