# CPT to SFT Training Plan

## Stage 1: CPT

### Confirmed inputs

- Base model:
  - `/ssd1/lisongyang/models/Qwen3.6-27B-ms-test`
- CPT train corpus:
  - `/ssd1/lisongyang/data/cryoagent_cpt_papers/papers_cpt_train.jsonl`
- CPT eval corpus:
  - `/ssd1/lisongyang/data/cryoagent_cpt_papers/papers_cpt_eval.jsonl`
- Corpus scale:
  - `661` papers
  - `22423` chunks
  - `1` failed paper

### Recommended first CPT run

- Slurm file:
  - `/home/lisongyang/cryoagent/submit_cpt_lora_fsdp_h20.slurm`
- Output directory:
  - `/ssd1/lisongyang/outputs/cryoagent-cpt-lora-h20-v1`
- GPUs:
  - `2 x H20`
- Precision:
  - `bf16`
- Sequence length:
  - `4096`
- Epochs:
  - `1`

### Goal

Produce a stable CPT LoRA adapter that can be merged into the base model for the next-stage SFT run.

## Stage 2: CPT -> SFT

### Confirmed SFT baseline data

- SFT train/eval root:
  - `/ssd1/lisongyang/data/cryoagent_sft_v2_no_workflow`
- Current baseline adapter:
  - `/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow`

### New handoff mechanism

The SFT script now supports:

- `--init-adapter-path`

Behavior:

1. load the raw base model from `--model-path`
2. load and merge the CPT adapter from `--init-adapter-path`
3. attach a fresh LoRA adapter for SFT training

This keeps the workflow simple and avoids a manual merged-checkpoint step.

### Recommended Stage-2 launch file

- `/home/lisongyang/cryoagent/submit_lora_sft_fsdp_h20_v2_no_workflow_from_cpt.slurm`

Recommended environment variables:

```bash
MODEL_PATH=/ssd1/lisongyang/models/Qwen3.6-27B-ms-test
INIT_ADAPTER_PATH=/ssd1/lisongyang/outputs/cryoagent-cpt-lora-h20-v1
OUTPUT_DIR=/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow-from-cpt
```

## Stage 3: Validation

After Stage 2, compare against the current SFT-only baseline using:

- JSON validity checks
- decision accuracy evaluation
- representative output export
- optional MCP-side closed-loop smoke checks

### Main comparison

- `Base -> SFT`
- `Base -> CPT -> SFT`

### Primary question

Does CPT improve:

- no-workflow decision quality
- exact action matching
- parameter selection quality
- robustness on longer or harder examples
