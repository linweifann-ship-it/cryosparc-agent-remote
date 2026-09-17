# Runs local or OpenAI-compatible models and converts text output into V2 decisions.
import copy
import json
import os
import re
import time
from http.client import RemoteDisconnected
from pathlib import Path
from urllib import error, request
from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a CryoSPARC workflow decision model. Return exactly one valid JSON "
    "object. Do not include markdown, comments, explanations, or thinking text."
)


def build_workflow_decision_prompt(
    model_input: Dict[str, Any],
    visual_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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
            "available, return request_input or stop. When particle diameter is "
            "missing, prefer a forward Blob Picker with a deliberately broad "
            "exploratory diameter range; use request_input only when no "
            "scientifically reasonable range can be proposed."
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
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]
    if visual_context:
        artifact_names = (
            ("pick_qc_dashboard", "contact_sheet")
            if visual_context.get("kind") == "pick_inspection"
            else ("contact_sheet",)
        )
        artifacts = {name: visual_context.get(name) or {} for name in artifact_names}
        attachment_status = {
            name: (
                "available"
                if isinstance(artifact.get("data_url"), str) and artifact["data_url"]
                else "unavailable"
            )
            for name, artifact in artifacts.items()
        }
        visual_text = {
            "visual_instruction": (
                "The attached image is a class-average contact sheet. Each tile is labelled "
                "class_id=<integer>. Inspect particle quality, structural consistency, "
                "noise, and view diversity. Use the labels exactly in selected_templates."
            ) if visual_context.get("kind") != "pick_inspection" else (
                "Review the Pick QC dashboard before the micrograph contact sheet. The dashboard "
                "contains an Exposure Plot and observed NCC Score × Power Score density, not a "
                "threshold recommendation. Use only threshold fields exposed by the candidate schema."
            ),
            "visual_context": {
                key: value for key, value in visual_context.items()
                if key not in {"contact_sheet", "pick_qc_dashboard"}
            },
            "visual_attachment_status": attachment_status,
            "visual_attachment_fallback": (
                "Make the decision from the structured state without requesting human input solely "
                "for an unavailable image."
            ),
        }
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": json.dumps(visual_text, ensure_ascii=False)},
        ]
        for artifact in artifacts.values():
            image_url = artifact.get("data_url") if isinstance(artifact, dict) else None
            if isinstance(image_url, str) and image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append({"role": "user", "content": content})
    return messages


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
    messages: List[Dict[str, str]],
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
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible endpoint with bounded transient-error retries."""
    if not api_base:
        raise ValueError("api_base is required for the OpenAI-compatible backend.")
    if not api_key:
        raise ValueError("api_key is required for the OpenAI-compatible backend.")
    if not model_name:
        raise ValueError("model_name is required for the OpenAI-compatible backend.")

    base_payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if tools:
        base_payload["tools"] = tools
    if tool_choice:
        base_payload["tool_choice"] = tool_choice

    cache_enabled = bool(prompt_cache_key or prompt_cache_options)

    def build_payload(include_cache: bool) -> Dict[str, Any]:
        payload = dict(base_payload)
        if include_cache:
            if prompt_cache_key:
                payload["prompt_cache_key"] = prompt_cache_key
            if prompt_cache_options:
                payload["prompt_cache_options"] = prompt_cache_options
        return payload

    def remove_explicit_cache_breakpoints(value: Any) -> Any:
        if isinstance(value, list):
            return [remove_explicit_cache_breakpoints(item) for item in value]
        if isinstance(value, dict):
            return {
                key: remove_explicit_cache_breakpoints(item)
                for key, item in value.items()
                if key != "prompt_cache_breakpoint"
            }
        return value

    endpoint = api_base.rstrip("/") + "/chat/completions"
    retryable_http_codes = {408, 429, 500, 502, 503, 504}
    # Some compatible gateways occasionally return a body-less 403 while the
    # upstream route is being refreshed. Retry that transient form once, but
    # never retry explicit authentication or quota failures.
    bodyless_forbidden_retries = 0
    cache_fallback_used = False
    attempts = 0
    while True:
        attempts += 1
        payload = build_payload(cache_enabled)
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
            break
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            cache_error = (
                cache_enabled
                and not cache_fallback_used
                and (
                    "prompt_cache_breakpoint" in error_body
                    or "prompt_cache_options" in error_body
                    or "prompt cache" in error_body.lower()
                )
            )
            if cache_error or (
                cache_enabled
                and not cache_fallback_used
                and exc.code == 403
                and not error_body.strip()
            ):
                # Providers are not consistent about exposing cache support in
                # their OpenAI-compatible surface. Retry once without optional
                # cache extensions; implicit prefix caching can still apply.
                cache_enabled = False
                cache_fallback_used = True
                base_payload["messages"] = remove_explicit_cache_breakpoints(
                    copy.deepcopy(base_payload["messages"])
                )
                continue
            if exc.code == 403 and not error_body.strip():
                if bodyless_forbidden_retries < 1:
                    bodyless_forbidden_retries += 1
                    time.sleep(retry_backoff_seconds * 2)
                    continue
            if exc.code not in retryable_http_codes or attempts > max_retries:
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: {error_body}"
                ) from exc
        except (error.URLError, RemoteDisconnected, ConnectionResetError, TimeoutError) as exc:
            error_body = str(exc)
            if attempts > max_retries:
                raise RuntimeError(
                    f"OpenAI-compatible request failed to reach {endpoint}: {error_body}"
                ) from exc
        time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))

    parsed = json.loads(raw_response)
    try:
        message = parsed["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI-compatible response did not contain choices[0].message."
        ) from exc

    content = message.get("content")
    raw_text = normalize_message_content(content) if content is not None else ""
    return {
        "endpoint": endpoint,
        "request_payload": payload,
        "raw_response": parsed,
        "raw_text": raw_text,
        "assistant_message": message,
        "tool_calls": message.get("tool_calls") or [],
        "attempts": attempts,
        "cache_fallback_used": cache_fallback_used,
    }


def resolve_api_key(
    explicit_api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Resolve an API key without requiring secrets in project files or argv."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    env_value = os.environ.get(api_key_env, "").strip()
    if env_value:
        return env_value
    env_file = Path(
        os.getenv("CRYOAGENT_API_ENV_FILE", "~/.config/cryoagent/api.env")
    ).expanduser()
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, value = stripped.partition("=")
            if separator and name.strip() == api_key_env:
                value = value.strip().strip("\"'")
                if value:
                    return value
    raise ValueError(
        f"API key not provided. Set --api-key, export {api_key_env}, "
        f"or configure {env_file}."
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
