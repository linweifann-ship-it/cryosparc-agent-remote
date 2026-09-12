#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test loading a local Qwen checkpoint.")
    parser.add_argument(
        "--model-path",
        default="/ssd1/lisongyang/models/Qwen3.6-27B-ms-test",
        help="Local path to the downloaded model directory.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only validate config, tokenizer, and shard metadata without instantiating model weights.",
    )
    return parser.parse_args()


def summarize_shards(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    shard_names = sorted(set(data["weight_map"].values()))
    shard_bytes = {name: (model_path / name).stat().st_size for name in shard_names}
    return {
        "index_path": str(index_path),
        "shard_count": len(shard_names),
        "mapped_entries": len(data["weight_map"]),
        "metadata_total_size": int(data.get("metadata", {}).get("total_size", 0)),
        "local_total_size": sum(shard_bytes.values()),
        "largest_shard": max(shard_bytes, key=shard_bytes.get),
    }


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    shard_summary = summarize_shards(model_path)
    print(f"Model directory: {model_path}")
    print(
        "Shard summary: "
        f"{shard_summary['shard_count']} shards, "
        f"{shard_summary['mapped_entries']} mapped tensors, "
        f"local_total_size={shard_summary['local_total_size']}, "
        f"index_total_size={shard_summary['metadata_total_size']}"
    )

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    print(
        "Config loaded: "
        f"model_type={config.model_type}, "
        f"hidden_size={getattr(config, 'hidden_size', 'n/a')}, "
        f"num_hidden_layers={getattr(config, 'num_hidden_layers', 'n/a')}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    sample = tokenizer("CryoSPARC workflow decision test.", return_tensors="pt")
    print(
        "Tokenizer loaded: "
        f"vocab_size={len(tokenizer)}, "
        f"input_ids_shape={tuple(sample['input_ids'].shape)}"
    )

    if args.skip_model:
        print("Skipping full weight load because --skip-model was set.")
        return

    # Load the complete checkpoint on CPU so this test is independent of GPU visibility.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    total_params = sum(param.numel() for param in model.parameters())
    embed_shape = tuple(model.get_input_embeddings().weight.shape)
    print(
        "Model loaded: "
        f"class={model.__class__.__name__}, "
        f"dtype={next(model.parameters()).dtype}, "
        f"total_params={total_params}, "
        f"embed_shape={embed_shape}"
    )


if __name__ == "__main__":
    main()
