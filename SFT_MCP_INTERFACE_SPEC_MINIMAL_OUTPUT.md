# CryoAgent Minimal Output Schema

## Goal

This schema is the new preferred model-facing output contract for SFT.
The model should predict only the decision semantics:

- `decision_type`
- `selected_actions[].job_type`
- `selected_actions[].parameters`

The model should not predict runtime-specific `workflow_node_id` values.
Those identifiers should be resolved later by the MCP server or executor.

## Output JSON

```json
{
  "schema_version": "3.0",
  "decision_type": "forward",
  "selected_actions": [
    {
      "job_type": "extract_micrographs_multi",
      "parameters": {
        "compute_num_gpus": 4,
        "box_size_pix": 400
      }
    }
  ]
}
```

## Field Definitions

- `schema_version`
  - fixed string: `3.0`
- `decision_type`
  - one of: `forward`, `branch`, `stop`
- `selected_actions`
  - list of actions to execute next
  - for `forward`, usually one action
  - for `branch`, multiple parallel actions
  - for `stop`, empty list

## Action Item Schema

```json
{
  "job_type": "blob_picker_gpu",
  "parameters": {
    "diameter": 200
  }
}
```

Only these two fields are model-facing:

- `job_type`
- `parameters`

Do not ask the model to emit:

- `workflow_node_id`
- `action_id`
- `action_type`
- `rollback_target`
- `branch_plan`
- free-form explanation text

## Why This Change

This design removes noise from inconsistent workflow node numbering across exported datasets.
The model focuses on the real decision target:

- what to do next
- whether to branch
- which parameters to use

The MCP server should later map the chosen `job_type` and `parameters` onto concrete runtime jobs or new workflow branches.

## Examples

### Forward

```json
{
  "schema_version": "3.0",
  "decision_type": "forward",
  "selected_actions": [
    {
      "job_type": "template_picker_gpu",
      "parameters": {
        "diameter": 200
      }
    }
  ]
}
```

### Branch

```json
{
  "schema_version": "3.0",
  "decision_type": "branch",
  "selected_actions": [
    {
      "job_type": "blob_picker_gpu",
      "parameters": {
        "diameter": 150
      }
    },
    {
      "job_type": "blob_picker_gpu",
      "parameters": {
        "diameter": 200
      }
    },
    {
      "job_type": "blob_picker_gpu",
      "parameters": {
        "diameter": 300
      }
    }
  ]
}
```

### Stop

```json
{
  "schema_version": "3.0",
  "decision_type": "stop",
  "selected_actions": []
}
```
