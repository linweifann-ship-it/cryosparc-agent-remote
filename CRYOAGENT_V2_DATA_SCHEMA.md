# CryoAgent V2 Data Schema

## Goal

V2 schema aligns the SFT task with the intended product behavior:

- input: dataset-level information + current workflow state
- output: next workflow action(s) + parameters

Compared with the current V1 pipeline, V2 shifts the task away from:

- explicit candidate-action selection
- parameter-template copying

and toward:

- dataset-aware decision making
- state-conditioned next-step generation

## Source Data

### Dataset-level sources

- `workflow_label.json`
  - path: `/hdd1/huangjianhua/agent/data/workflow/workflow_label.json`
  - fields:
    - `EMPIAR_ID`
    - `resolution`
    - `input`
    - `types`
    - `num_of_maps`

- `emd-xxxx.xml`
  - path: `/hdd1/huangjianhua/agent/data/experiment/emd-xxxx.xml`
  - fields used for:
    - `emdb_id`
    - `abstract`
    - optional reference summary

### Workflow-level sources

- `workflow/<EMPIAR_ID>.json`
  - path: `/hdd1/huangjianhua/agent/data/workflow`
  - fields used for:
    - known workflow steps
    - workflow node parameters
    - workflow node order / dependencies

- `Log/<EMPIAR_ID>/jXX.json`
  - path: `/hdd1/huangjianhua/agent/data/Log`
  - fields used for:
    - current node state
    - runtime status
    - output metrics
    - image references

## Design Principles

### 1. Single model, two input states

We do not split into two models.

The same model should handle:

- known-workflow datasets
- unknown-workflow datasets

The distinction is represented in the input payload:

- if known workflow exists:
  - `known_workflow_steps` is populated
- if not:
  - `known_workflow_steps` is `null`

### 2. Known workflow means full reference workflow is available

For known datasets, the full ordered workflow can be given to the model as a reference.

This is intentionally different from the earlier “history only” interpretation.

Current product assumption:

- if upstream MCP server can retrieve a known workflow template for this dataset,
  it may provide the full ordered steps
- if not, it passes `null`

### 3. Current state remains the decision anchor

Even when known workflow exists, the model should still rely on:

- current node status
- current outputs / metrics
- last step summary

The known workflow is reference context, not a substitute for state.

## Top-level Input Schema

```json
{
  "schema_version": "2.0",
  "task_type": "workflow_decision",
  "dataset_info": {},
  "current_state": {}
}
```

## `dataset_info`

```json
{
  "empiar_id": "EMPIAR-12099",
  "emdb_id": "EMD-50426",
  "resolution": [3.0],
  "input_type": "particle",
  "macromolecules_type": "protein",
  "num_of_maps": 1,
  "abstract": "Primary citation title... Sample: ... Resolution: ...",
  "known_workflow_steps": []
}
```

### Field Definitions

- `empiar_id`
  - string
  - normalized form like `EMPIAR-12099`

- `emdb_id`
  - string or `null`
  - normalized form like `EMD-50426`

- `resolution`
  - array
  - copied from `workflow_label.json`

- `input_type`
  - string or `null`
  - copied from `workflow_label.json`

- `macromolecules_type`
  - string or `null`
  - copied from `workflow_label.json`

- `num_of_maps`
  - integer or `null`
  - copied from `workflow_label.json`

- `abstract`
  - string or `null`
  - compact textual summary derived from EMDB XML citation / sample / resolution info

- `known_workflow_steps`
  - array or `null`
  - if dataset has retrievable known workflow:
    - full ordered workflow step list
  - if not:
    - `null`

## `known_workflow_steps[]`

```json
{
  "step_index": 0,
  "node_id": "J2028",
  "action": "import_volumes",
  "title": "",
  "description": "",
  "upstream_node_ids": [],
  "parameter_template": {
    "volume_blob_path": "/hdd1/msai/db/emdb/emd_50426.map"
  }
}
```

### Field Definitions

- `step_index`
  - integer
  - stable sorted order within the reference workflow

- `node_id`
  - logical workflow node ID

- `action`
  - workflow job type

- `title`
  - optional workflow-node title

- `description`
  - optional workflow-node description

- `upstream_node_ids`
  - ordered parent node IDs

- `parameter_template`
  - workflow-declared parameter template for the step
  - this is reference workflow information, not runtime current-state parameters

## `current_state`

```json
{
  "last_node_id": "J2280",
  "last_action": "patch_ctf_estimation_multi",
  "last_node_status": "completed",
  "last_node_info": {}
}
```

### Field Definitions

- `last_node_id`
  - string or `null`
  - previous logical workflow node ID

- `last_action`
  - string or `null`
  - previous job type

- `last_node_status`
  - string
  - recommended values:
    - `not_started`
    - `running`
    - `completed`
    - `failed`
    - `waiting`

- `last_node_info`
  - structured summary of the previous node runtime state

## `last_node_info`

```json
{
  "job_type": "patch_ctf_estimation_multi",
  "job_uid": "J2280",
  "job_title": "New Job J2280",
  "project_uid": "P1",
  "status": "completed",
  "timestamps": {
    "created_at": "...",
    "started_at": "...",
    "completed_at": "..."
  },
  "inputs": {
    "groups": []
  },
  "parameters": {
    "compute_num_gpus": 4
  },
  "outputs": {
    "groups": []
  },
  "metrics": {
    "micrograph_count": 509,
    "total_runtime_seconds": 1234.5
  },
  "runtime": {
    "lane": "default",
    "worker_hostname": "H20b0",
    "allocated_gpu": "[1, 0, 2, 3]",
    "allocated_ssd": "False"
  },
  "evidence_text": [
    "Job ready to run",
    "Job will process this many micrographs: 509"
  ],
  "warning_lines": [],
  "image_refs": {}
}
```

### Main Sections

- `timestamps`
  - created / started / completed

- `inputs`
  - compact copy of input groups

- `parameters`
  - flattened runtime parameters from `params_spec`

- `outputs.groups`
  - summarized output-group objects

- `metrics`
  - decision-oriented numeric metrics

- `runtime`
  - lane / worker / cpu / gpu / ssd / working directory

- `evidence_text`
  - short textual evidence lines

- `warning_lines`
  - warning-only subset

- `image_refs`
  - image metadata and categorized references

## Output Schema

V2 keeps the current output schema unchanged.

```json
{
  "schema_version": "1.0",
  "decision_type": "forward",
  "selected_actions": [],
  "rollback_target": null,
  "branch_plan": null,
  "reason": "",
  "confidence": 0.0,
  "risk_flags": [],
  "evidence": []
}
```

## MCP Server Retrieval Logic

Recommended responsibility split:

- MCP server:
  - determines dataset identity
  - retrieves known workflow if available
  - passes full `known_workflow_steps`
  - passes `null` if workflow not found

- model:
  - reads `dataset_info`
  - reads `current_state`
  - decides next action and parameters

This avoids asking the model to perform repository lookup implicitly.

## Image Handling

The local Qwen3.6-27B base model supports native image input:

- has `vision_config`
- has `image_token_id`
- README advertises `image-text-to-text`

However, current V2 preprocessing is still text-first.

Short-term recommendation:

- store image references in `last_node_info.image_refs`
- derive textual/structured metrics from logs first

Long-term recommendation:

- add true multimodal SFT samples that include selected PNGs as visual input

## Recommended Sample Construction

For a workflow with `N` decision points:

- create `N + 1` samples
  - one sample per next-step decision
  - one final `stop` sample

Each sample contains:

- same `dataset_info`
- current `last_node_*` state for that decision point
- target next action(s)

## Known Risks

### 1. Full known workflow can simplify the task

Providing full known workflow steps may make the task easier for in-domain datasets.

This is acceptable for the intended “known workflow replay” mode,
but should be evaluated separately from unknown-workflow exploration.

### 2. Unknown-workflow evaluation must be explicit

For unknown-workflow testing:

- hide known workflow retrieval at MCP layer
- set `known_workflow_steps = null`
- evaluate exploration behavior separately

## Immediate Next Steps

1. implement `last_node_info` extractor
2. build a V2 batch preprocessor
3. generate a small V2 sample batch
4. manually inspect 10-20 samples
5. decide whether `known_workflow_steps` should be full workflow or pruned workflow in production


## Missing Input and Exploratory Decisions

The V2 decision layer supports `request_input` when a required fact such as `particle_diameter_A` is unavailable. This is a planning response only and must not create a CryoSPARC job.

For an explicitly exploratory Blob Picker, use `forward` with `action: "blob_picker_gpu"`, provide `diameter` and `diameter_max`, and mark the decision with `risk_flags: ["exploratory_parameter_range"]`.
