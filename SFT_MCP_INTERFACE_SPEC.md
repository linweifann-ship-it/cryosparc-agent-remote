# CryoSPARC Agent SFT / MCP Interface Spec

## Goal

This document defines the contract between:

- the SFT-trained model, which outputs structured JSON decisions
- the downstream MCP server, which validates the JSON and calls cryoSPARC tools

The model is **not** responsible for directly invoking tools.
The model is responsible for:

- reading the current workflow state
- choosing the next action
- returning strict JSON in a fixed schema

The MCP server is responsible for:

- validating the JSON
- mapping `workflow_node_id` / `job_type` to concrete tool calls
- creating jobs in cryoSPARC
- handling retries, execution monitoring, and tool-side failures


## Scope

The current training objective is a **workflow decision task**.

Model input includes:

- completed workflow nodes
- current node state
- workflow labels
- candidate next actions

Model output includes:

- whether to move forward, roll back, branch, or stop
- which node or nodes to run next
- parameters for each next node
- evidence and risk flags for downstream use

This design assumes:

- most current data are successful workflow runs
- a smaller number of real or synthetic rollback/branch cases will be added later
- the model should operate in a **constrained action space**, not open-ended planning


## Terminology

Two job ID systems must be kept separate.

- `workflow_node_id`
  - logical node ID inside a workflow template
  - examples: `J1`, `J2`, `J10`
- `actual_job_id`
  - real cryoSPARC runtime job ID
  - examples: `J350`, `J1028`, `J1453`

The model should primarily reason over `workflow_node_id`.
The MCP server may use `actual_job_id` for traceability and execution bookkeeping.


## Data Sources

The training pipeline may use:

- workflow definitions such as [empiar-10025-workflow.json](/Users/lisongyang/test/cryofold/data_process/empiar-10025-workflow.json:1)
- workflow labels such as [workflow_label.json](/Users/lisongyang/test/cryofold/data_process/workflow_label.json:1)
- parsed PDF event logs
- ZIP job logs containing `job_document`, `job_log`, `command_core`, `command_rtp`, and related files

Recommended canonical sources by role:

- workflow structure:
  - `workflow.json`
- workflow-level metadata/labels:
  - `workflow_label.json`
- runtime state and job evidence:
  - parsed PDF/ZIP logs


## Workflow Labels

`workflow_label.json` should be normalized before use.
At the moment, it appears to be JSON with a leading stray `.` character and should be cleaned during preprocessing.

Recommended normalized label schema:

```json
{
  "dataset_id": "EMPIAR-10025",
  "empiar_id": 10025,
  "input_type": "micrograph",
  "sample_type": "protein",
  "target_resolution": [8.2],
  "num_of_maps": 1
}
```

Suggested mapping from current label file:

- `EMPIAR_ID` -> `empiar_id`
- `input` -> `input_type`
- `types` -> `sample_type`
- `resolution` -> `target_resolution`
- `num_of_maps` -> `num_of_maps`

The preprocessing layer should also derive:

- `dataset_id`
  - format: `EMPIAR-<EMPIAR_ID>`


## Model Input Schema

The model input should be a single JSON object.
The MCP server should construct this input from canonical workflow state.

### Required top-level keys

```json
{
  "schema_version": "1.0",
  "task_type": "workflow_decision",
  "dataset": {},
  "workflow": {},
  "current_state": {},
  "candidate_actions": [],
  "constraints": {}
}
```

### Input Schema Details

```json
{
  "schema_version": "1.0",
  "task_type": "workflow_decision",
  "dataset": {
    "dataset_id": "EMPIAR-10025",
    "empiar_id": 10025,
    "input_type": "micrograph",
    "sample_type": "protein",
    "target_resolution": [8.2],
    "num_of_maps": 1
  },
  "workflow": {
    "workflow_id": "wf_empiar_10025",
    "workflow_title": "EMPIAR-10025",
    "workflow_version": "1.0.0",
    "current_workflow_node_id": "J6",
    "completed_nodes": [
      {
        "workflow_node_id": "J1",
        "job_type": "import_volumes",
        "actual_job_id": "J350",
        "status": "completed"
      },
      {
        "workflow_node_id": "J2",
        "job_type": "import_micrographs",
        "actual_job_id": "J1028",
        "status": "completed"
      }
    ],
    "available_upstream_outputs": [
      "J5.particles",
      "J5.micrographs"
    ]
  },
  "current_state": {
    "workflow_node_id": "J6",
    "job_type": "inspect_picks_v2",
    "actual_job_id": "J1036",
    "status": "completed",
    "project": {},
    "job": {},
    "inputs": {},
    "parameters": {},
    "outputs": {},
    "derived_metrics": {},
    "quality_flags": [
      "particle_quality_uncertain"
    ],
    "evidence": [
      "1472266 particles included, 374397 particles excluded."
    ]
  },
  "candidate_actions": [
    {
      "action_id": "forward_J7",
      "action_type": "forward",
      "workflow_node_id": "J7",
      "job_type": "extract_micrographs_multi",
      "allowed_inputs": [
        "J6.micrographs",
        "J6.particles"
      ],
      "parameter_template": {
        "compute_num_gpus": 4,
        "box_size_pix": 400
      }
    },
    {
      "action_id": "rollback_J5",
      "action_type": "rollback",
      "workflow_node_id": "J5",
      "job_type": "template_picker_gpu",
      "allowed_inputs": [
        "J3.templates",
        "J4.exposures"
      ],
      "parameter_template": {
        "diameter": 200
      }
    }
  ],
  "constraints": {
    "max_branches": 3,
    "must_choose_from_candidates": true,
    "return_json_only": true
  }
}
```


## Model Output Schema

The model output must be a single valid JSON object with no surrounding prose.

### Required top-level keys

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

### Field Definitions

- `schema_version`
  - fixed string, currently `1.0`
- `decision_type`
  - one of:
    - `forward`
    - `rollback`
    - `branch`
    - `stop`
- `selected_actions`
  - list of one or more actions chosen from `candidate_actions`
- `rollback_target`
  - required when `decision_type == "rollback"`
- `branch_plan`
  - required when `decision_type == "branch"`
- `reason`
  - short natural-language explanation
- `confidence`
  - float in `[0, 1]`
- `risk_flags`
  - list of machine-readable warning tags
- `evidence`
  - list of short supporting observations from the state


## Action Object Schema

Every item in `selected_actions` must follow this schema:

```json
{
  "action_id": "forward_J7",
  "action_type": "forward",
  "workflow_node_id": "J7",
  "job_type": "extract_micrographs_multi",
  "parameters": {
    "compute_num_gpus": 4,
    "box_size_pix": 400
  }
}
```

Rules:

- `action_id` should match a provided candidate action whenever available
- `workflow_node_id` must be from the candidate set
- `job_type` must match the selected workflow node
- `parameters` must be valid JSON and conform to the node parameter template


## Rollback Target Schema

When `decision_type` is `rollback`, `rollback_target` must be populated.

```json
{
  "workflow_node_id": "J5",
  "job_type": "template_picker_gpu",
  "reason_code": "poor_particle_picking"
}
```

Recommended `reason_code` values:

- `low_ctf_quality`
- `poor_particle_picking`
- `bad_2d_classes`
- `refine_underperforming`
- `parameter_too_strict`
- `parameter_too_loose`
- `input_mismatch`
- `resource_issue`


## Branch Plan Schema

When `decision_type` is `branch`, `branch_plan` must be populated.

```json
{
  "branch_type": "parallel_hypothesis_test",
  "max_parallel_branches": 3,
  "notes": "Run multiple refinement hypotheses in parallel."
}
```

Recommended `branch_type` values:

- `parameter_sweep`
- `symmetry_search`
- `parallel_refinement`
- `alternate_pick_strategy`
- `alternate_cleanup_strategy`
- `parallel_hypothesis_test`


## Output Examples

### Example A: Normal Forward Progression

```json
{
  "schema_version": "1.0",
  "decision_type": "forward",
  "selected_actions": [
    {
      "action_id": "forward_J7",
      "action_type": "forward",
      "workflow_node_id": "J7",
      "job_type": "extract_micrographs_multi",
      "parameters": {
        "compute_num_gpus": 4,
        "box_size_pix": 400
      }
    }
  ],
  "rollback_target": null,
  "branch_plan": null,
  "reason": "Particle inspection passed and particles are ready for extraction.",
  "confidence": 0.94,
  "risk_flags": [],
  "evidence": [
    "Current node inspect_picks_v2 completed successfully.",
    "Selected particles are available for downstream extraction."
  ]
}
```

### Example B: Rollback Decision

```json
{
  "schema_version": "1.0",
  "decision_type": "rollback",
  "selected_actions": [],
  "rollback_target": {
    "workflow_node_id": "J5",
    "job_type": "template_picker_gpu",
    "reason_code": "poor_particle_picking"
  },
  "branch_plan": null,
  "reason": "Particle quality is not sufficient for extraction and picking should be adjusted.",
  "confidence": 0.83,
  "risk_flags": [
    "particle_quality_uncertain"
  ],
  "evidence": [
    "Too many particles were excluded after inspection.",
    "Inspection metrics suggest thresholds or picking quality need adjustment."
  ]
}
```

### Example C: Multi-Path Branch

```json
{
  "schema_version": "1.0",
  "decision_type": "branch",
  "selected_actions": [
    {
      "action_id": "branch_J10",
      "action_type": "branch",
      "workflow_node_id": "J10",
      "job_type": "homo_refine_new",
      "parameters": {
        "refine_symmetry": "D7",
        "refine_defocus_refine": false,
        "refine_ctf_global_refine": false
      }
    },
    {
      "action_id": "branch_J11",
      "action_type": "branch",
      "workflow_node_id": "J11",
      "job_type": "homo_refine_new",
      "parameters": {
        "refine_symmetry": "D7",
        "refine_defocus_refine": true,
        "refine_ctf_global_refine": false
      }
    },
    {
      "action_id": "branch_J12",
      "action_type": "branch",
      "workflow_node_id": "J12",
      "job_type": "homo_refine_new",
      "parameters": {
        "refine_symmetry": "D7",
        "refine_defocus_refine": false,
        "refine_ctf_global_refine": true
      }
    }
  ],
  "rollback_target": null,
  "branch_plan": {
    "branch_type": "parallel_refinement",
    "max_parallel_branches": 3,
    "notes": "Evaluate multiple refinement variants from the same selected particle set."
  },
  "reason": "The current state supports parallel refinement hypotheses.",
  "confidence": 0.88,
  "risk_flags": [
    "multiple_refine_hypotheses_recommended"
  ],
  "evidence": [
    "Selected particles are available.",
    "Multiple valid homo_refine_new variants exist in the workflow template."
  ]
}
```

### Example D: Stop

```json
{
  "schema_version": "1.0",
  "decision_type": "stop",
  "selected_actions": [],
  "rollback_target": null,
  "branch_plan": null,
  "reason": "Workflow is complete and no further action is required.",
  "confidence": 0.98,
  "risk_flags": [],
  "evidence": [
    "All terminal workflow nodes have completed successfully."
  ]
}
```


## MCP Server Responsibilities

The MCP server should treat model output as a **proposal**, not an executable command.

Minimum validation steps:

- verify JSON is parseable
- verify `schema_version`
- verify `decision_type`
- verify chosen actions exist in `candidate_actions`
- verify `workflow_node_id` and `job_type` match workflow definition
- verify parameters are type-compatible with the target node
- verify branch count does not exceed `constraints.max_branches`
- verify rollback target is allowed by policy

Recommended server behavior by decision type:

- `forward`
  - create exactly one next job unless policy allows multiple
- `rollback`
  - do not auto-delete prior jobs
  - create a rerun or alternate job from the rollback target
- `branch`
  - create multiple jobs in parallel
  - attach a shared branch group ID for traceability
- `stop`
  - mark workflow as complete or await human review


## Training Recommendations

Recommended SFT target is this exact output schema.

Suggested training sample types:

- `real_success`
  - standard forward progression
- `real_branch`
  - true multi-path exploration
- `real_failure`
  - authentic rollback or blocked states
- `synthetic_counterfactual`
  - rule-based variants derived from real workflows
- `synthetic_branch`
  - generated branch decisions from valid templates

Recommended metadata for each sample:

```json
{
  "data_source": "real_success",
  "workflow_decision_label": "forward",
  "has_synthetic_perturbation": false
}
```


## Guardrails

The model should not:

- invent new workflow nodes that are not in `candidate_actions`
- emit tool call syntax
- emit prose outside JSON
- delete or mutate historical jobs directly
- assume unsupported rollback paths

The MCP server should not:

- execute model output without validation
- assume `confidence` is calibrated enough to skip safety checks
- allow unconstrained branch fan-out


## Versioning

Start with:

- input schema version: `1.0`
- output schema version: `1.0`

If fields change later:

- bump `schema_version`
- keep backward compatibility in the MCP validation layer when practical


## Open Decisions For Team Alignment

The following items should be confirmed with the MCP server owner:

- whether `candidate_actions` are always provided by the server
- whether rollback actions should appear in `candidate_actions` or be separately enumerated
- whether `confidence` is required for execution
- whether `reason` and `evidence` are logged only, or also shown to users
- whether branch creation should happen automatically or require approval
- whether parameter validation is strict or allows server-side default filling


## Recommended Next Step

Implement the preprocessing pipeline so that each workflow instance produces:

1. canonical workflow state records
2. candidate action sets
3. target decision JSON in this schema

That dataset can then be converted into chat-format SFT samples where:

- user input = serialized input schema JSON
- assistant output = serialized output schema JSON
