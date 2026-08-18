# Model provider clients for validation-only Schema V24 benchmark runs.
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from model_direct_runner import run_qwen_lora_model
from v24_contract import V24_OUTPUT_JSON_SCHEMA


@dataclass
class ModelClientConfig:
    provider: str
    model_label: str
    temperature: float = 0.0
    seed: int | None = None
    api_timeout_seconds: int = 120
    base_model: str | None = None
    adapter: str | None = None
    openai_model: str | None = None
    qwen_model: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_native_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    dashscope_api_mode: str = "compatible"
    max_new_tokens: int = 768
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    qwen_response_format: str = "json_object"
    finetune_checkpoint: str | None = None


class ModelClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        request_id: str,
    ) -> dict[str, Any]:
        ...


def create_model_client(config: ModelClientConfig) -> ModelClient:
    if config.provider == "local":
        return LocalQwenClient(config)
    if config.provider == "openai":
        return OpenAIResponsesClient(config)
    if config.provider == "dashscope_qwen":
        return DashScopeQwenClient(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


class LocalQwenClient:
    def __init__(self, config: ModelClientConfig):
        self.config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        request_id: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            generation = run_qwen_lora_model(
                messages=messages,
                base_model_path=str(self.config.base_model),
                adapter_path=self.config.adapter,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
            )
            return model_result(
                success=True,
                provider="local",
                model_label=self.config.model_label,
                exact_model_id=self.config.adapter or self.config.base_model,
                raw_text=generation.get("raw_text", ""),
                sanitized_response={"request_id": request_id},
                usage=None,
                latency=time.monotonic() - started,
                request_id=request_id,
                finetune_checkpoint=self.config.finetune_checkpoint or self.config.adapter,
            )
        except Exception as exc:
            return model_error(
                provider="local",
                model_label=self.config.model_label,
                exact_model_id=self.config.adapter or self.config.base_model,
                request_id=request_id,
                latency=time.monotonic() - started,
                error=exc,
                finetune_checkpoint=self.config.finetune_checkpoint or self.config.adapter,
            )


class OpenAIResponsesClient:
    def __init__(self, config: ModelClientConfig):
        self.config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        request_id: str,
    ) -> dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return missing_key_result("openai", self.config, request_id, "OPENAI_API_KEY")
        body = {
            "model": self.config.openai_model,
            "input": messages,
            "temperature": self.config.temperature,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "schema_v24_minimal_v3",
                    "schema": V24_OUTPUT_JSON_SCHEMA,
                    "strict": True,
                }
            },
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        return post_json(
            url="https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            body=body,
            provider="openai",
            model_label=self.config.model_label,
            exact_model_id=self.config.openai_model,
            request_id=request_id,
            timeout=self.config.api_timeout_seconds,
            extract_text=extract_openai_response_text,
            finetune_checkpoint=self.config.finetune_checkpoint,
        )


class DashScopeQwenClient:
    def __init__(self, config: ModelClientConfig):
        self.config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        request_id: str,
    ) -> dict[str, Any]:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return missing_key_result(
                "dashscope_qwen",
                self.config,
                request_id,
                "DASHSCOPE_API_KEY",
            )
        if self.config.dashscope_api_mode == "native":
            body = {
                "model": self.config.qwen_model,
                "input": {"messages": messages},
                "parameters": {
                    "temperature": self.config.temperature,
                    "result_format": "message",
                    "response_format": {"type": self.config.qwen_response_format},
                },
            }
            if self.config.seed is not None:
                body["parameters"]["seed"] = self.config.seed
            return post_json(
                url=self.config.dashscope_native_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-DashScope-SSE": "disable",
                },
                body=body,
                provider="dashscope_qwen",
                model_label=self.config.model_label,
                exact_model_id=self.config.qwen_model,
                request_id=request_id,
                timeout=self.config.api_timeout_seconds,
                extract_text=extract_dashscope_native_text,
                finetune_checkpoint=self.config.finetune_checkpoint,
            )

        response_format: dict[str, Any]
        if self.config.qwen_response_format == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema_v24_minimal_v3",
                    "schema": V24_OUTPUT_JSON_SCHEMA,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}
        body = {
            "model": self.config.qwen_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": response_format,
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        return post_json(
            url=f"{self.config.dashscope_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            body=body,
            provider="dashscope_qwen",
            model_label=self.config.model_label,
            exact_model_id=self.config.qwen_model,
            request_id=request_id,
            timeout=self.config.api_timeout_seconds,
            extract_text=extract_chat_completion_text,
            finetune_checkpoint=self.config.finetune_checkpoint,
        )


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    provider: str,
    model_label: str,
    exact_model_id: str | None,
    request_id: str,
    timeout: int,
    extract_text,
    finetune_checkpoint: str | None,
) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            **headers,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_text = extract_text(payload)
        return model_result(
            success=True,
            provider=provider,
            model_label=model_label,
            exact_model_id=payload.get("model") or exact_model_id,
            raw_text=raw_text,
            sanitized_response=sanitize_response(payload),
            usage=payload.get("usage"),
            latency=time.monotonic() - started,
            request_id=payload.get("id") or request_id,
            finetune_checkpoint=finetune_checkpoint,
        )
    except Exception as exc:
        return model_error(
            provider=provider,
            model_label=model_label,
            exact_model_id=exact_model_id,
            request_id=request_id,
            latency=time.monotonic() - started,
            error=exc,
            finetune_checkpoint=finetune_checkpoint,
        )


def extract_openai_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def extract_dashscope_native_text(payload: dict[str, Any]) -> str:
    output = payload.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content") or ""
    text = output.get("text")
    return text if isinstance(text, str) else ""


def sanitize_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep response metadata without hidden reasoning or bulky content."""
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "created": payload.get("created"),
        "usage": payload.get("usage"),
        "status": payload.get("status"),
        "error": payload.get("error"),
    }


def missing_key_result(
    provider: str,
    config: ModelClientConfig,
    request_id: str,
    env_name: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "provider": provider,
        "model_label": config.model_label,
        "exact_model_id": config.openai_model or config.qwen_model or config.adapter,
        "raw_text": "",
        "sanitized_response": None,
        "usage": None,
        "latency": 0.0,
        "request_id": request_id,
        "error": {
            "category": "api_failure",
            "type": "MissingAPIKey",
            "message": f"{env_name} is not set.",
        },
        "finetune_checkpoint": config.finetune_checkpoint,
    }


def model_result(
    success: bool,
    provider: str,
    model_label: str,
    exact_model_id: str | None,
    raw_text: str,
    sanitized_response: dict[str, Any] | None,
    usage: dict[str, Any] | None,
    latency: float,
    request_id: str,
    finetune_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "success": success,
        "provider": provider,
        "model_label": model_label,
        "exact_model_id": exact_model_id,
        "raw_text": raw_text,
        "sanitized_response": sanitized_response,
        "usage": usage,
        "latency": latency,
        "request_id": request_id,
        "error": None,
        "finetune_checkpoint": finetune_checkpoint,
    }


def model_error(
    provider: str,
    model_label: str,
    exact_model_id: str | None,
    request_id: str,
    latency: float,
    error: Exception,
    finetune_checkpoint: str | None,
) -> dict[str, Any]:
    status = None
    message = str(error)
    if isinstance(error, urllib.error.HTTPError):
        status = error.code
        try:
            message = error.read().decode("utf-8")
        except Exception:
            message = str(error)
    return {
        "success": False,
        "provider": provider,
        "model_label": model_label,
        "exact_model_id": exact_model_id,
        "raw_text": "",
        "sanitized_response": {"http_status": status} if status else None,
        "usage": None,
        "latency": latency,
        "request_id": request_id,
        "error": {
            "category": "api_failure",
            "type": type(error).__name__,
            "message": message,
            "http_status": status,
        },
        "finetune_checkpoint": finetune_checkpoint,
    }


def write_model_client_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
