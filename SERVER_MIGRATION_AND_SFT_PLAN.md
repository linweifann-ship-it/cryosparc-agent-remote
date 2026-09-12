# CryoSPARC Agent Server Migration And SFT Plan

## Purpose

This document is for moving the current local data-processing pipeline to a server and for setting up the next-stage SFT training framework.

It covers:

- what has already been implemented locally
- how to run full-dataset preprocessing on a server
- what training data files are produced
- how to organize SFT training around the current workflow-decision task
- what to defer until more rollback / branch data are available


## Current Project Scope

The current project goal is to train a model that can act as a **cryoSPARC workflow decision module**.

The model does not directly call cryoSPARC tools.
Instead:

- the model reads structured workflow state
- the model outputs strict JSON
- a downstream MCP server validates that JSON and performs tool invocation

The current SFT target is:

- input:
  - completed workflow nodes
  - current node state
  - workflow label metadata
  - candidate next actions
- output:
  - `forward` / `rollback` / `branch` / `stop`
  - selected next node or nodes
  - parameters for selected nodes
  - optional evidence / risk flags

The interface contract for this model-server boundary is documented in:

- [SFT_MCP_INTERFACE_SPEC.md](/Users/lisongyang/test/cryofold/data_process/SFT_MCP_INTERFACE_SPEC.md:1)


## Current Local Files

### Core preprocessing scripts

- [prepare_cryosparc_sft_data.py](/Users/lisongyang/test/cryofold/data_process/prepare_cryosparc_sft_data.py:1)
  - single-job log parsing
  - supports cryoSPARC PDF event logs
  - extracts `project`, `job`, `inputs`, `parameters`, `outputs`, `derived_metrics`

- [prepare_cryosparc_workflow_sft_data.py](/Users/lisongyang/test/cryofold/data_process/prepare_cryosparc_workflow_sft_data.py:1)
  - workflow-level preprocessing
  - consumes workflow JSON, workflow labels, and JobLog directories
  - produces node-level records, decision records, and chat-format SFT data

### Interface / design docs

- [SFT_MCP_INTERFACE_SPEC.md](/Users/lisongyang/test/cryofold/data_process/SFT_MCP_INTERFACE_SPEC.md:1)
  - model input/output schema for MCP server alignment

### Example data

- [empiar-10025-workflow.json](/Users/lisongyang/test/cryofold/data_process/empiar-10025-workflow.json:1)
- [workflow_label.json](/Users/lisongyang/test/cryofold/data_process/workflow_label.json:1)
- [EMPIAR-10025-JobLog](/Users/lisongyang/test/cryofold/data_process/EMPIAR-10025-JobLog)


## What The Current Pipeline Does

### 1. Single-job parsing

For one cryoSPARC job log, the pipeline extracts:

- `project`
- `job`
- `inputs`
- `parameters`
- `outputs`
- `derived_metrics`

This is the basic structured representation of one cryoSPARC job.

### 2. Workflow-level reconstruction

For one workflow instance, the pipeline combines:

- workflow template JSON
- workflow label metadata
- per-node cryoSPARC job logs

and reconstructs:

- workflow DAG dependencies
- workflow node parameters
- actual cryoSPARC runtime job IDs
- per-node runtime state
- transition-level decision samples

### 3. Decision-sample generation

The workflow pipeline converts one successful workflow into multiple SFT samples.

Each step becomes a decision record:

- current state
- candidate next nodes
- target selected action(s)

This expands one workflow into multiple training samples instead of only one sample.


## Important Data Conventions

### Two job ID systems must stay separate

- `workflow_node_id`
  - logical node in workflow template
  - example: `J1`, `J2`, `J10`

- `actual_job_id`
  - real cryoSPARC runtime job
  - example: `J350`, `J1028`, `J1453`

Do not merge these two concepts in preprocessing or training.

### Current naming convention for batch matching

The current workflow batch script assumes:

- workflow file:
  - `empiar-xxxxx-workflow.json`
- corresponding log directory:
  - `EMPIAR-xxxxx-JobLog`

This matching rule is already implemented in the workflow preprocessing script.

### Workflow label file cleanup

`workflow_label.json` currently contains a leading stray `.` character.
The preprocessing script already strips this before JSON parsing.


## Server Directory Recommendation

Recommended server workspace:

```text
/data/cryofold/
  code/
    prepare_cryosparc_sft_data.py
    prepare_cryosparc_workflow_sft_data.py
    SFT_MCP_INTERFACE_SPEC.md
    SERVER_MIGRATION_AND_SFT_PLAN.md
  data/
    workflow_label.json
    workflows/
      empiar-10025-workflow.json
      empiar-10026-workflow.json
      ...
    joblogs/
      EMPIAR-10025-JobLog/
      EMPIAR-10026-JobLog/
      ...
  outputs/
    preprocessing/
    training/
```


## Server Environment Setup

Recommended Python version:

- Python 3.9 or newer

Recommended setup:

```bash
cd /data/cryofold/code
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pypdf pdfplumber
```

Notes:

- `pdfplumber` is preferred for PDF extraction because some cryoSPARC PDFs are slow or unstable with `pypdf` alone
- `prepare_cryosparc_sft_data.py` already tries `pdfplumber` first and falls back to `pypdf`


## Running Full-Dataset Preprocessing

### Mode A: explicit single workflow + log directory

Use this when debugging one dataset.

```bash
python3 prepare_cryosparc_workflow_sft_data.py \
  --workflow-path /data/cryofold/data/workflows/empiar-10025-workflow.json \
  --log-dir /data/cryofold/data/joblogs/EMPIAR-10025-JobLog \
  --workflow-labels /data/cryofold/data/workflow_label.json \
  --node-records-jsonl /data/cryofold/outputs/preprocessing/workflow_node_records.jsonl \
  --decision-records-jsonl /data/cryofold/outputs/preprocessing/workflow_decision_records.jsonl \
  --sft-jsonl /data/cryofold/outputs/preprocessing/workflow_sft_data.jsonl
```

### Mode B: batch auto-alignment

Use this for the full dataset.

```bash
python3 prepare_cryosparc_workflow_sft_data.py \
  --workflow-root /data/cryofold/data/workflows \
  --log-root /data/cryofold/data/joblogs \
  --workflow-labels /data/cryofold/data/workflow_label.json \
  --node-records-jsonl /data/cryofold/outputs/preprocessing/workflow_node_records.jsonl \
  --decision-records-jsonl /data/cryofold/outputs/preprocessing/workflow_decision_records.jsonl \
  --sft-jsonl /data/cryofold/outputs/preprocessing/workflow_sft_data.jsonl
```

### Expected outputs

The script produces:

- `workflow_node_records.jsonl`
- `workflow_decision_records.jsonl`
- `workflow_sft_data.jsonl`

It also reports:

- `workflow_count`
- `node_record_count`
- `decision_record_count`
- `sft_record_count`
- `missing_log_dirs`

`missing_log_dirs` is important for full-dataset auditing.


## Output File Roles

### 1. `workflow_node_records.jsonl`

This is the canonical node-level dataset.

Recommended uses:

- data auditing
- feature analysis
- future rollback / branch augmentation
- non-chat training tasks

Main fields include:

- `dataset_id`
- `workflow_id`
- `workflow_node_id`
- `workflow_job_type`
- `workflow_upstream_nodes`
- `workflow_parameters`
- `actual_job_id`
- `project`
- `job`
- `inputs`
- `parameters`
- `outputs`
- `derived_metrics`
- `evidence`
- `risk_flags`

### 2. `workflow_decision_records.jsonl`

This is the structured decision-level dataset.

Each row contains:

- `model_input`
- `model_output`

This file is the best source for:

- strict JSON supervised training
- offline validation
- rule-based augmentation

### 3. `workflow_sft_data.jsonl`

This is chat-format SFT data.

Each row contains:

- `id`
- `dataset_id`
- `workflow_id`
- `messages`
- `metadata`

This is the most convenient file for most open-source chat SFT frameworks.


## Training Target Definition

The current recommended SFT task is:

- input:
  - current workflow state JSON
- output:
  - decision JSON

This is a **constrained workflow policy task**, not open-ended planning.

### Why this target is appropriate now

Current data characteristics:

- about 200+ complete workflow cases
- mostly successful workflows
- very limited rollback / branch exception data

Therefore, the model can learn:

- standard progression patterns
- node selection
- parameter instantiation
- valid structured output

But the model cannot yet fully learn:

- robust failure recovery
- broad exception handling
- open-ended replanning


## Recommended Training Data Strategy

### Phase 1: train on successful workflow decisions

Use the current `workflow_sft_data.jsonl` to train the model to:

- output strict JSON
- choose valid next actions from candidate actions
- emit correct parameters for selected nodes
- handle `forward`, `branch`, and `stop`

### Phase 2: add rollback / exception data

When more rollback / branch cases are collected, extend the same schema to include:

- `rollback`
- branch alternatives with different reasons
- synthetic counterfactual cases derived from real workflows

### Phase 3: optional multi-task training

Later, training can be split into:

- policy decision head
- parameter generation head
- auxiliary confidence / evidence prediction

For now, a single JSON output task is sufficient.


## Recommended Server-Side Training Framework

### Base model choice

Use an open-source instruction-tuned model with strong JSON-following behavior.

Typical candidates:

- 7B to 14B class model for first iteration
- prefer a model already good at structured output and tool-use style reasoning

Initial recommendation:

- start with a smaller model for pipeline verification
- then scale to a larger model after data and evaluation stabilize

### Data format for training

Use `workflow_sft_data.jsonl` directly when the training framework expects chat messages.

If the framework expects prompt/response pairs, convert:

- prompt = `messages[0] + messages[1]`
- response = `messages[2]`

### Training recipe recommendation

Start simple:

- supervised fine-tuning only
- no RL at the first stage
- no online tool interaction at the first stage

Recommended first-pass setup:

- LoRA or QLoRA
- chat-format SFT
- max sequence length chosen from actual prompt length statistics
- strict validation on JSON format correctness

### Suggested experiment progression

1. Run preprocessing on the full dataset
2. Audit counts and missing logs
3. Split train / dev / test by workflow, not by decision row
4. Run a small LoRA baseline
5. Evaluate JSON validity and decision accuracy
6. Add synthetic rollback / branch data later


## Suggested Train / Dev / Test Split

Do not split by individual decision sample only.
Split by workflow instance.

Recommended:

- train: 80%
- dev: 10%
- test: 10%

Reason:

- decisions from the same workflow are highly correlated
- splitting at the row level would leak workflow structure across sets


## Evaluation Recommendations

The first evaluation target should not be only token loss.
Track at least these metrics:

### 1. JSON validity

- is the output parseable JSON
- does it contain all required keys

### 2. Decision type accuracy

- correct `forward` / `branch` / `stop`
- later also `rollback`

### 3. Node selection accuracy

- whether selected node IDs match target

### 4. Parameter exact-match or field-level match

- whether generated parameters match the target node configuration

### 5. Constraint adherence

- selected nodes must come from `candidate_actions`
- branch count must stay within `max_branches`

### 6. Optional calibration

- compare `confidence` against actual correctness


## Training Framework Implementation Suggestion

Recommended server repo layout:

```text
/data/cryofold/training/
  configs/
    sft_lora.yaml
  scripts/
    train_sft.py
    eval_sft.py
    convert_jsonl.py
  data/
    workflow_sft_train.jsonl
    workflow_sft_dev.jsonl
    workflow_sft_test.jsonl
  outputs/
    checkpoints/
    logs/
```

Suggested framework choices:

- Hugging Face Transformers + PEFT
- TRL SFTTrainer
- or another in-house training framework that supports chat-format JSONL

The important part is not the framework brand.
The important part is preserving:

- chat message format
- exact JSON output objective
- workflow-level split


## Suggested Training Loop

1. Preprocess all workflows into `workflow_sft_data.jsonl`
2. Split by workflow ID into train / dev / test
3. Fine-tune with LoRA
4. Evaluate:
   - JSON validity
   - action selection accuracy
   - parameter accuracy
5. Save best checkpoint
6. Run offline inference on held-out workflows
7. Compare model output with MCP schema validator


## Future Data Augmentation Direction

Because current real data are mostly successful workflows, future improvements should focus on:

- real rollback cases
- real branch cases
- synthetic counterfactual rollback cases
- synthetic branch exploration cases

These synthetic samples should still use the same input/output schema, so no retraining pipeline redesign is needed later.


## Known Limitations Of The Current Pipeline

### 1. Success-biased dataset

Current main dataset teaches:

- standard flow
- standard parameters
- standard branching from successful reference workflows

It does not yet teach robust failure recovery.

### 2. PDF parsing is heuristic

PDF logs are parsed heuristically.
ZIP logs are generally richer and more structured.

Where both are available, ZIP is the more valuable source.

### 3. Some `evidence` fields may still be verbose

Current evidence extraction is already trimmed, but further normalization may still help before final large-scale training.


## Recommended Immediate Next Steps On Server

1. Copy code and data into the recommended server layout
2. Create Python environment and install dependencies
3. Run one explicit single-workflow preprocessing test
4. Run full batch preprocessing
5. Audit:
   - `missing_log_dirs`
   - output counts
   - random sample quality
6. Freeze one preprocessed snapshot for reproducibility
7. Build train / dev / test split by workflow
8. Start first LoRA SFT baseline


## Minimal Command Checklist

### Environment

```bash
cd /data/cryofold/code
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pypdf pdfplumber
```

### Single-workflow preprocessing smoke test

```bash
python3 prepare_cryosparc_workflow_sft_data.py \
  --workflow-path /data/cryofold/data/workflows/empiar-10025-workflow.json \
  --log-dir /data/cryofold/data/joblogs/EMPIAR-10025-JobLog \
  --workflow-labels /data/cryofold/data/workflow_label.json
```

### Full preprocessing

```bash
python3 prepare_cryosparc_workflow_sft_data.py \
  --workflow-root /data/cryofold/data/workflows \
  --log-root /data/cryofold/data/joblogs \
  --workflow-labels /data/cryofold/data/workflow_label.json \
  --node-records-jsonl /data/cryofold/outputs/preprocessing/workflow_node_records.jsonl \
  --decision-records-jsonl /data/cryofold/outputs/preprocessing/workflow_decision_records.jsonl \
  --sft-jsonl /data/cryofold/outputs/preprocessing/workflow_sft_data.jsonl
```


## Handoff Notes

For MCP-side alignment:

- keep using the schema defined in [SFT_MCP_INTERFACE_SPEC.md](/Users/lisongyang/test/cryofold/data_process/SFT_MCP_INTERFACE_SPEC.md:1)
- do not let the model directly call tools
- validate all model output before cryoSPARC execution

For model-side alignment:

- the immediate training objective is strict JSON decision generation
- `workflow_sft_data.jsonl` is the main training artifact
- future rollback / branch expansion should reuse the same schema
