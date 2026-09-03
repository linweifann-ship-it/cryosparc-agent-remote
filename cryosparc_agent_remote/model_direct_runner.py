# Runs local or OpenAI-compatible models and converts text output into V2 decisions.
import json
import os
import re
import time
from urllib import error, request
from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a CryoSPARC workflow decision model. Return exactly one valid JSON "
    "object. Do not include markdown, comments, explanations, or thinking text."
)


def build_workflow_decision_prompt(model_input: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for the direct model closed-loop test."""
    output_contract = {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop | request_input",
        "action": "CryoSPARC job type for forward/branch decisions, or omit for stop.",
        "parameters": "Only parameters that should override defaults.",
        "reason": "Short reason for the decision.",
        "confidence": "Number from 0.0 to 1.0.",
        "risk_flags": [],
        "evidence": [],
        "requested_inputs": "Required only for request_input, e.g. [\"particle_diameter_A\"].",
    }
    user_content = {
        "instruction": (
            "Choose the next workflow action from the current CryoSPARC state. "
            "Only select a job type present in model_input.candidate_actions; "
            "do not invent connections or job types. If no safe candidate is "
            "available, return request_input or stop."
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
            {
                "schema_version": "2.0",
                "decision_type": "request_input",
                "requested_inputs": ["particle_diameter_A"],
                "reason": "Blob picking requires a particle diameter that is not available.",
                "confidence": 0.95,
                "risk_flags": ["needs_human_input"],
                "evidence": ["No reliable particle diameter is present in the dataset context."],
            },
            {
                "schema_version": "2.0",
                "decision_type": "forward",
                "action": "blob_picker_gpu",
                "parameters": {"diameter": 150, "diameter_max": 250},
                "reason": "Use a deliberately broad exploratory diameter range.",
                "confidence": 0.65,
                "risk_flags": ["exploratory_parameter_range"],
                "evidence": ["Particle size is uncertain; the range is explicitly exploratory."],
            },
        ],
    }
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def render_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
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
    messages: List[Dict[str, str]],
    base_model_path: str,
    adapter_path: Optional[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
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


def run_openai_compatible_model(
    messages: List[Dict[str, Any]],
    api_base: str,
    api_key: str,
    model_name: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible endpoint with bounded transient-error retries."""
    if not api_base:
        raise ValueError("api_base is required for the OpenAI-compatible backend.")
    if not api_key:
        raise ValueError("api_key is required for the OpenAI-compatible backend.")
    if not model_name:
        raise ValueError("model_name is required for the OpenAI-compatible backend.")

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_options:
        payload["prompt_cache_options"] = prompt_cache_options
    endpoint = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    retryable_http_codes = {408, 429, 500, 502, 503, 504}
    attempts = 0
    request_started = time.perf_counter()
    while True:
        attempts += 1
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
            break
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable_http_codes or attempts > max_retries:
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: {error_body}"
                ) from exc
        except error.URLError as exc:
            error_body = str(exc)
            if attempts > max_retries:
                raise RuntimeError(
                    f"OpenAI-compatible request failed to reach {endpoint}: {error_body}"
                ) from exc
        time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))

    parsed = json.loads(raw_response)
    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI-compatible response did not contain choices[0].message.content."
        ) from exc

    raw_text = normalize_message_content(content)
    return {
        "endpoint": endpoint,
        "request_payload": payload,
        "exact_serialized_request": body.decode("utf-8"),
        "request_chars": len(body),
        "latency_ms": round((time.perf_counter() - request_started) * 1000, 3),
        "raw_response": parsed,
        "usage": extract_usage_summary(parsed),
        "raw_text": raw_text,
        "attempts": attempts,
    }


def extract_usage_summary(response: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize token/cache counters across OpenAI-compatible response shapes."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {
            "present": False,
            "cached_tokens": None,
            "cache_field_path": None,
        }

    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    cached_tokens = None
    cache_field_path = None
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        cached_tokens = prompt_details.get("cached_tokens")
        cache_field_path = "usage.prompt_tokens_details.cached_tokens"
    elif isinstance(input_details, dict) and "cached_tokens" in input_details:
        cached_tokens = input_details.get("cached_tokens")
        cache_field_path = "usage.input_tokens_details.cached_tokens"

    return {
        "present": True,
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_field_path": cache_field_path,
    }


def resolve_api_key(
    explicit_api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Resolve an API key from an explicit value or an environment variable."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    env_value = os.environ.get(api_key_env, "").strip()
    if env_value:
        return env_value
    raise ValueError(
        f"API key not provided. Set --api-key or export {api_key_env}."
    )


def normalize_message_content(content: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise ValueError(
        f"Unsupported message content type from API response: {type(content).__name__}"
    )


def normalize_optional_path(path: Optional[str]) -> Optional[str]:
    """Treat empty strings and common sentinel values as no adapter."""
    if path is None:
        return None
    stripped = path.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def parse_model_decision_text(text: str) -> Dict[str, Any]:
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
