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
