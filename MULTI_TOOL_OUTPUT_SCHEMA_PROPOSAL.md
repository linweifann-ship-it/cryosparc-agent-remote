# Multi-Tool Output Schema Proposal

## Goal

Extend the current CryoSPARC-oriented decision JSON into a tool-agnostic schema that can support:

- cryoSPARC
- RELION
- Warp
- ChimeraX / PHENIX / Coot / ISOLDE
- future EM-processing or structure-modeling tools

The key idea is:

- keep the **decision layer** generic
- move tool-specific details into the **execution layer**

In other words, the model should decide:

- what to do next
- on which object
- with what intent

while MCP should decide:

- which tool implements that action
- how to translate parameters
- how to validate and execute it safely

## Why the current schema is CryoSPARC-specific

Current action objects contain fields like:

- `workflow_node_id`
- `job_type`
- CryoSPARC-style parameters

Example:

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

This works for CryoSPARC, but it assumes:

- jobs are represented as `Jxx`
- actions are software-job names
- parameters follow one tool's API

That will not scale well to other systems.

## Design Principle

Split the schema into three layers:

1. `decision layer`
   - generic model output
2. `tool binding layer`
   - maps generic action to one or more tool implementations
3. `execution layer`
   - MCP-side validation, translation, and dispatch

The model should only emit the **decision layer**.

## Proposed V3 top-level output schema

```json
{
  "schema_version": "3.0",
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

This top level can remain almost unchanged from `1.0`.

The main change is inside `selected_actions`.

## Proposed V3 action schema

```json
{
  "action_id": "action_001",
  "action_type": "forward",
  "operation": "particle_picking",
  "target_ref": "state.last_micrographs",
  "tool_preference": [
    "cryosparc",
    "warp"
  ],
  "intent": {
    "goal": "generate_particle_candidates",
    "strategy": "blob_based_picking"
  },
  "parameters": {
    "particle_diameter_angstrom": 180,
    "minimum_interparticle_distance_angstrom": 120
  },
  "expected_outputs": [
    "particles",
    "picking_diagnostics"
  ]
}
```

## New field meanings

- `operation`
  - generic action category
  - examples:
    - `import_movies`
    - `motion_correction`
    - `ctf_estimation`
    - `particle_picking`
    - `particle_extraction`
    - `two_d_classification`
    - `class_selection`
    - `ab_initio_reconstruction`
    - `homogeneous_refinement`
    - `heterogeneous_refinement`
    - `atomic_model_building`
    - `real_space_refinement`
    - `validation`

- `target_ref`
  - what object in the current state the action should operate on
  - examples:
    - `state.last_micrographs`
    - `state.selected_particles`
    - `state.current_volume`
    - `state.current_atomic_model`

- `tool_preference`
  - optional ranked list of preferred tools
  - examples:
    - `["cryosparc", "warp"]`
    - `["phenix", "coot"]`

- `intent`
  - semantic explanation of why this action is being proposed
  - useful when multiple tools can do similar things

- `parameters`
  - generic semantic parameters, not tool-native arguments

- `expected_outputs`
  - optional expected artifacts for downstream MCP validation

## Example mappings

### CryoSPARC-specific current action

Current:

```json
{
  "job_type": "blob_picker_gpu",
  "parameters": {
    "diameter": 180,
    "diameter_max": 220
  }
}
```

Generalized:

```json
{
  "operation": "particle_picking",
  "target_ref": "state.last_micrographs",
  "tool_preference": ["cryosparc"],
  "intent": {
    "goal": "generate_particle_candidates",
    "strategy": "blob_based_picking"
  },
  "parameters": {
    "particle_diameter_angstrom": 180,
    "particle_diameter_upper_angstrom": 220
  }
}
```

### StructAgent-style model-building action

```json
{
  "action_id": "action_014",
  "action_type": "forward",
  "operation": "real_space_refinement",
  "target_ref": "state.current_atomic_model",
  "tool_preference": [
    "phenix",
    "refmac5"
  ],
  "intent": {
    "goal": "improve_model_map_agreement",
    "strategy": "restrained_refinement"
  },
  "parameters": {
    "resolution_limit_angstrom": 3.8,
    "apply_secondary_structure_restraints": true
  },
  "expected_outputs": [
    "refined_model",
    "validation_report"
  ]
}
```

## What should move out of the model output

These fields should no longer be required in the model output:

- `workflow_node_id`
- `job_type`
- tool-native parameter names like:
  - `compute_num_gpus`
  - `class2D_K`
  - `refine_ctf_global_refine`

These should instead be resolved by MCP adapters.

## MCP-side adapter design

Each supported tool should have an adapter like:

- `CryoSPARCAdapter`
- `RelionAdapter`
- `WarpAdapter`
- `PhenixAdapter`
- `CootAdapter`

Each adapter should define:

- supported `operation` values
- mapping from generic parameters to tool-native parameters
- input/output object mapping
- safety checks
- execution method

Example:

```python
class CryoSPARCAdapter:
    supports = {"particle_picking", "two_d_classification", "homogeneous_refinement"}

    def resolve(self, action, state):
        ...
```

## Recommended input-side extension

To support multi-tool output, the **input** should also expose a tool-agnostic state.

Current state fields like:

- `last_action`
- `last_node_id`

should gradually be complemented by generic fields like:

- `last_operation`
- `available_artifacts`
- `artifact_refs`
- `tool_history`

Example:

```json
{
  "current_state": {
    "last_operation": "ctf_estimation",
    "last_status": "completed",
    "available_artifacts": [
      {
        "artifact_id": "art_001",
        "artifact_type": "micrographs",
        "producer_tool": "cryosparc",
        "producer_ref": "J43.exposures"
      }
    ],
    "tool_history": [
      {
        "tool": "cryosparc",
        "operation": "motion_correction",
        "status": "completed"
      }
    ]
  }
}
```

## Migration strategy

Recommended practical migration:

### Phase 1
- Keep current CryoSPARC schema working.
- Add optional new fields:
  - `operation`
  - `target_ref`
  - `tool_preference`

### Phase 2
- MCP resolves `job_type` from `operation`.
- Train the model to emit both:
  - generic semantic fields
  - CryoSPARC-compatible fallback fields

### Phase 3
- Remove tool-native requirements from model output.
- Make the model fully tool-agnostic.
- Let MCP server own all tool binding.

## Practical recommendation for this project

For the next iteration, I would recommend:

1. Keep the current JSON shape at the top level.
2. Replace `job_type` with a more general `operation` field.
3. Add `tool_preference` as an optional ranked list.
4. Treat `workflow_node_id` as MCP-internal rather than model-facing.
5. Define a small controlled vocabulary of generic EM operations first.

This gives us a path where:

- current CryoSPARC experiments keep working
- future expansion to RELION / Warp / StructAgent-style downstream tools becomes possible
- the model learns scientific workflow reasoning rather than one software's API
