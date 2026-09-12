# Current Progress Results

This note summarizes the experimental results that are currently solid enough to present in the **Current Progress** section of the group meeting PPT.

## 1. Model Output Quality

### 1.1 Representative successful generation
- In exported examples, the model can already produce executable JSON decisions with structured fields such as `decision_type`, `selected_actions`, `parameters`, `branch_plan`, `reason`, `confidence`, and `evidence`.
- A representative case is `EMPIAR-10059:v2_step:000`, where the model correctly generated a parallel `branch` decision for:
- `import_volumes`
- `import_particles`
- The output was valid JSON and matched the expected high-level structure.

### 1.2 Representative failure mode
- A representative failure appears in `EMPIAR-12146:step:000`.
- The model generated a long multi-branch output, but the JSON was truncated before completion.
- The parse error was:
- `JSONDecodeError('Unterminated string starting at: line 89 column 7 (char 2555)')`
- This suggests that long branch outputs remain a practical failure case even when the decision direction is reasonable.

## 2. Offline Accuracy Results

### 2.1 Known-workflow / strong-context setting
Source:
- `/home/lisongyang/cryoagent/logs/eval_decision_accuracy_1104389/summary.json`

Result summary:
- `sample_count`: `152`
- `valid_json_rate`: `0.9868`
- `decision_type_accuracy`: `0.9868`
- `selected_action_set_accuracy`: `0.9868`
- `selected_action_parameters_accuracy`: `0.9868`
- `core_decision_exact_match_accuracy`: `0.9868`

Interpretation:
- In a setting with strong workflow guidance, the model can achieve very high exact-match performance.
- This can be presented as an upper-bound or easier-setting result rather than the main target result.

### 2.2 Earlier failed prompt variant
Source:
- `/home/lisongyang/cryoagent/logs/eval_decision_accuracy_1110445/summary.json`

Result summary:
- `sample_count`: `152`
- `valid_json_rate`: `1.0000`
- `decision_type_accuracy`: `0.1184`
- `selected_action_set_accuracy`: `0.1184`
- `selected_action_parameters_accuracy`: `0.1184`
- `core_decision_exact_match_accuracy`: `0.1184`

Interpretation:
- This run shows that output formatting alone is not enough.
- The model can produce valid JSON while still collapsing to poor decisions.
- This result is useful as a negative control and helps motivate later schema and prompt improvements.

### 2.3 Current no-workflow validation result
Source:
- `/home/lisongyang/cryoagent/logs/cryoagent-eval-acc-1126149.out`

Result summary:
- `sample_count`: `137`
- `valid_json_count`: `136`
- `valid_json_rate`: `0.9927`
- `decision_type_accuracy`: `0.9635`
- `selected_action_count_accuracy`: `0.9416`
- `selected_action_set_accuracy`: `0.5547`
- `selected_action_parameters_accuracy`: `0.3942`
- `core_decision_exact_match_accuracy`: `0.3942`

Interpretation:
- This is the most important current model result for presentation.
- It shows that the model is already strong at:
- producing valid JSON
- identifying the correct decision type
- estimating how many actions should be selected
- The main remaining weakness is exact action selection and parameter matching in the no-workflow setting.

## 3. JSON-Only Generation Check

Source:
- `/home/lisongyang/cryoagent/logs/cryoagent-json-check-1126099.out`

Observed behavior:
- The model produced valid JSON output for `EMPIAR-10059:v2_step:000`.
- The example correctly generated a `branch` decision with two actions.
- Confidence was `0.99`.

Interpretation:
- This supports the claim that the current inference setup can already generate structured model outputs suitable for downstream programmatic consumption.

## 4. MCP Server Experimental Results

## 4.1 Safe live smoke test
Source:
- `/ssd1/linweifan/cryosparc_agent/reports/2026-06-27_smoke_import_movies_live_report.md`

Observed result:
- MCP validated model-style JSON.
- MCP created a dry-run execution plan successfully.
- MCP created a real CryoSPARC `Import Movies` job: `J7`.
- The created job stayed in `building` state and was not queued for compute.
- Approval logic was already active for:
- high GPU count
- rollback
- interactive jobs

Interpretation:
- This verifies that the basic MCP-to-CryoSPARC tool chain works safely in live mode.

## 4.2 Closed-loop model -> MCP -> CryoSPARC chain
Source:
- `/ssd1/linweifan/cryosparc_agent/reports/model_closed_loop/interactions/interaction_summary.md`

Observed sequence:
- After completed `select_2D` job `J35`, the model predicted:
- `forward -> class_2D_new`
- parameters: `compute_num_gpus = 4`, `class2D_K = 50`
- MCP generated a plan on lane `g8m192_4090_slurm`.
- MCP created live job `J36`.
- `J36` completed successfully and produced expected outputs such as `particles` and `class_averages`.
- After `J36`, the model predicted:
- `forward -> select_2D`
- MCP recognized this as an interactive step.
- MCP created live job `J37`, then returned:
- `ready_for_model = false`
- `status_group = human_action_required`

Interpretation:
- This is a strong system result because it demonstrates:
- real job creation
- real job completion monitoring
- correct handoff back to the model
- correct interruption when a human-interactive step is reached

## 4.3 Additional live MCP job execution
Representative sources:
- `/ssd1/linweifan/cryosparc_agent/reports/model_closed_loop/xml_abi_w5/10_mcp_live_import_micrographs_full_params.json`
- `/ssd1/linweifan/cryosparc_agent/reports/model_closed_loop/xml_abi_w5/23_mcp_live_blob_picker.json`

Observed results:
- MCP created live `import_micrographs` job `J42` with explicit dataset parameters.
- MCP created live `blob_picker_gpu` job `J59` from completed `J43` outputs.
- The `blob_picker_gpu` job was queued on `g8m192_4090_slurm`.
- These runs also show that MCP can fall back to a generic execution path when the model proposes a valid job type that does not match an explicit internal candidate.

Interpretation:
- The MCP prototype is already more than a static interface layer.
- It can create and manage multiple real CryoSPARC job types under controlled execution logic.

## 5. Suggested PPT Version for Current Progress

Recommended points for one slide:
- `No-workflow validation`: `99.27%` valid JSON, `96.35%` decision-type accuracy, `39.42%` exact action+parameter match on `137` samples.
- `Known-workflow upper bound`: `98.68%` exact match on `152` samples.
- `MCP closed-loop`: real CryoSPARC jobs `J36`, `J42`, and `J59` created/monitored successfully; interactive `J37` correctly paused for human action.
- `Current interpretation`: high-level decision quality is already strong, while fine-grained action/parameter accuracy and long-output robustness still need improvement.
