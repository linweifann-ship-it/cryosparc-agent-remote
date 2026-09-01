# Generate raw model responses for the autonomous MCP closed-loop runner.
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_direct_runner import normalize_optional_path, render_chat_prompt, run_qwen_lora_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages-json-file")
    parser.add_argument("--output-json-file")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if not args.messages_json_file or not args.output_json_file:
        raise SystemExit("--messages-json-file and --output-json-file are required outside --worker mode.")
    messages = json.loads(Path(args.messages_json_file).read_text())
    generation = run_qwen_lora_model(
        messages=messages,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    output_path = Path(args.output_json_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(generation, indent=2, ensure_ascii=False, default=str)
    )


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = normalize_optional_path(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    write_worker_event(
        {
            "event": "model_loaded",
            "base_model": args.base_model,
            "adapter": adapter_path,
            "device": str(next(model.parameters()).device),
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "shutdown":
            write_worker_event({"event": "shutdown"})
            break
        messages = request["messages"]
        prompt = render_chat_prompt(tokenizer, messages)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(request.get("max_new_tokens", args.max_new_tokens)),
            "do_sample": float(request.get("temperature", args.temperature)) > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        temperature = float(request.get("temperature", args.temperature))
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        write_worker_event(
            {
                "event": "generation",
                "request_id": request.get("request_id"),
                "prompt": prompt,
                "raw_text": raw_text,
            }
        )


def write_worker_event(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
