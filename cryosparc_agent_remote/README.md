# CryoSPARC Agent Remote

Current version: **V2 Workflow Decision Loop**

`cryosparc_agent_remote` provides MCP tools for safely connecting an upstream
decision model to a CryoSPARC instance on `172.16.1.2`. The project currently
supports real workflow-state extraction, V2 model input generation, model
decision validation/adaptation, real child-job creation, GPU-lane submission,
internal job monitoring, and completed-job result packaging.

The default execution path is still conservative: `execute_model_decision` and
`execute_v2_model_decision` validate model output first and return a planned
execution while `dry_run=true`. Live execution is available for approved
forward actions and has been tested on the `g8m192_4090_slurm` lane.

## Architecture

The system has five layers:

1. **CryoSPARC access layer**
   - `cryosparc_client.py` creates authenticated `cryosparc-tools` clients.
   - `cryosparc_cli_tools.py` wraps CryoSPARC CLI commands such as status,
     version, GPU list, and worker tests.

2. **Workflow state layer**
   - Reads jobs from a CryoSPARC project/workspace.
   - Converts the workspace into a normalized DAG with nodes, edges, inputs,
     outputs, running nodes, and failed nodes.
   - Lives in `workflow_state.py`.

3. **Job metadata, execution, and monitoring layer**
   - `job_specs.py` stores small "job cards" for common CryoSPARC job types:
     editable parameters, GPU needs, interactive behavior, and approval needs.
   - `job_executor.py` converts validated actions into generic
     `workspace.create_job(job_type, connections, params)` plans or live jobs.
   - `job_result.py` keeps queue/running states internal and packages only
     completed/failed results for model-facing updates.

4. **Decision alignment layer**
   - Defines the upstream model output schema.
   - Supports `forward`, `rollback`, `branch`, and `stop`.
   - Validates action IDs, job types, workflow node IDs, parameter types,
     and parameter ranges against the supplied candidate actions.
   - `v2_decision_adapter.py` maps compact V2 model decisions back to the
     internal candidate-action executor.
   - Lives in `schemas.py`, `action_registry.py`, and
     `v2_decision_adapter.py`.

5. **Model input and MCP tool layer**
   - `model_input_builder.py` builds V2 model-facing payloads with
     `dataset_info` and `current_state`.
   - `known_workflow_retriever.py` fills `known_workflow_steps` from local
     workflow files when a dataset match is available; otherwise it returns
     `null`.
   - Exposes the project as MCP tools.
   - Lives in `cryosparc_mcp_server.py`.

## Server Run Command

Use the server-side conda environment that already has `mcp` and
`cryosparc-tools` installed:

```bash
cd /ssd1/linweifan/cryosparc_agent
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python cryosparc_mcp_server.py
```

## Direct Model Closed-Loop Test

Before the model API exists, run the model directly on the Hangzhou server:

```bash
cd /ssd1/linweifan/cryosparc_agent
env PYTHONPATH=/ssd1/linweifan/cryosparc_agent \
  /ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python \
  scripts/smoke_model_closed_loop.py \
  --project P2 \
  --workspace W3 \
  --current-node J8 \
  --dataset-json '{"empiar_id":"EMPIAR-10025","input_type":"micrographs"}'
```

Default behavior is dry-run: it builds the V2 MCP-to-model payload, calls the
local Qwen model with `enable_thinking=False`, parses the returned JSON,
adapts it to the internal candidate action, and returns the planned execution.
It does not create or queue a CryoSPARC job unless `--live` is added.

On this SLURM cluster, CryoSPARC API access is available on the login/master
host while model inference runs on GPU compute nodes. Use the split workflow
when running through SLURM:

```bash
# 1. Login node: build the V2 MCP-to-model payload from CryoSPARC.
env PYTHONPATH=/ssd1/linweifan/cryosparc_agent \
  /ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python \
  scripts/smoke_model_input_v2.py \
  --project P2 --workspace W3 --job J8 \
  --empiar-id EMPIAR-10025 \
  --input-type micrographs \
  --macromolecules-type ribosome \
  > reports/model_closed_loop/J8_v2_model_input.json

# 2. H20 GPU node: call the local model only.
srun -p g8m768 --gres=gpu:1 --time=00:45:00 \
  env PYTHONPATH=/ssd1/linweifan/cryosparc_agent \
  /ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python \
  scripts/smoke_direct_model_decision.py \
  --model-input-json-file reports/model_closed_loop/J8_v2_model_input.json \
  --decision-output-file reports/model_closed_loop/J8_model_decision.json \
  --report-output-file reports/model_closed_loop/J8_model_generation_report.json

# 3. Login node: adapt/validate/execute the saved decision.
env PYTHONPATH=/ssd1/linweifan/cryosparc_agent \
  /ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python \
  scripts/smoke_execute_v2_decision_file.py \
  --project P2 --workspace W3 --current-node J8 \
  --decision-json-file reports/model_closed_loop/J8_model_decision.json
```

Model paths used by default:

- Base model: `/ssd1/lisongyang/models/Qwen3.6-27B-ms-test`
- LoRA adapter: `/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow`

CryoSPARC paths used by the tools:

- Master CLI: `/ssd1/linweifan/cryosparc/cryosparc_master/bin/cryosparcm`
- Worker CLI: `/ssd1/linweifan/cryosparc/cryosparc_worker/bin/cryosparcw`
- Config: `/ssd1/linweifan/cryosparc/cryosparc_master/config.sh`

The `cryosparc-tools` client connects to `localhost:61000` and reads
`CRYOSPARC_LICENSE_ID` from `config.sh` if `CRYOSPARC_EMAIL` and
`CRYOSPARC_PASSWORD` are not set.

## Tools

### `get_cryosparc_status`

Checks whether CryoSPARC master services are running.

### `get_cryosparc_version`

Gets the installed CryoSPARC version.

### `get_cryosparc_worker_gpulist`

Runs:

```bash
/ssd1/linweifan/cryosparc/cryosparc_worker/bin/cryosparcw gpulist --format json
```

On the current SLURM cluster setup, running this on the master/login node may
return `CUDA_ERROR_NO_DEVICE`; that stderr is preserved in the structured
result.

### `test_cryosparc_workers`

Runs CryoSPARC worker validation jobs:

```bash
/ssd1/linweifan/cryosparc/cryosparc_master/bin/cryosparcm test workers PROJECT --test TEST
```

Inputs:

- `project_uid`: CryoSPARC project UID, for example `P1`.
- `test`: one of `launch`, `ssd`, `gpu`, or `all`. Default: `launch`.
- `target`: optional scheduler target, for example `h20_slurm`.
- `test_pytorch`: optional boolean. Default: `false`.
- `timeout`: command timeout in seconds. Default: `600`.

This tool creates CryoSPARC-side validation jobs in the specified project.

### `create_cryosparc_import_movies_job`

Creates an Import Movies job with `cryosparc-tools`.

Inputs:

- `project_uid`: CryoSPARC project UID, for example `P1`.
- `workspace_uid`: CryoSPARC workspace UID, for example `W1`.
- `blob_paths`: movie path glob, for example `/data/movies/*.tif`.
- `title`: optional job title. Default: `Import Movies`.
- `desc`: optional job description.
- `params`: optional extra Import Movies parameters.

This tool creates a real CryoSPARC job.

### `get_workflow_state`

Reads a CryoSPARC workspace and returns a normalized, read-only DAG snapshot.

Output includes:

- `nodes`
- `edges`
- `root_nodes`
- `terminal_nodes`
- `running_nodes`
- `failed_nodes`
- `node_mapping`

### `get_candidate_actions`

Returns candidate actions derived from the real CryoSPARC workflow state.

Each action includes:

- `action_id`
- `action_type`
- `workflow_node_id`
- `reference_job_uid`
- `job_type`
- `execution_mode`
- `required_inputs`
- `parameter_template`
- `default_parameters`

`workflow_node_id` uses the real CryoSPARC Job UID, for example `J8`. The
diagnostic logical label, when present, is separate and should not be used in
model decisions.

All generated actions currently remain `dry_run_only`.

GPU actions with `compute_num_gpus > 4` are marked for human approval in the
generated `execution_plan` with approval reason `high_gpu_count`.

### `get_supported_job_types`

Returns the job types that currently have explicit local metadata in
`job_specs.py`. Unknown job types can still be represented, but they default to
human approval before live execution.

### `validate_model_decision`

Validates an upstream model decision JSON against schema version `1.0` and a
candidate action set.

The validator checks:

- top-level schema fields and value ranges
- `decision_type` rules
- selected `action_id` membership
- `action_type`, `workflow_node_id`, and `job_type` consistency
- allowed parameters and parameter type/range constraints

This tool is validation-only. It does not create or enqueue CryoSPARC jobs.

### `execute_model_decision`

Validates a model decision and returns an execution plan.

Inputs:

- `decision`: upstream model decision JSON.
- `project_uid`, `workspace_uid`, `current_node_id`: optional live context used
  to regenerate candidate actions.
- `candidate_actions`: optional caller-supplied candidate action list.
- `dry_run`: optional boolean. Default: `true`.

Behavior:

- If validation fails, returns `execution_mode="validation_failed"`.
- If validation succeeds and `dry_run=true`, returns an `execution_plan`; it
  does not create jobs.
- If `dry_run=false`, approved forward actions can create and queue real
  CryoSPARC jobs through `job_executor.py`.

Example dry-run result:

```json
{
  "success": true,
  "dry_run": true,
  "execution_mode": "dry_run",
  "decision_type": "forward",
  "execution_plan": {
    "plan_id": "plan_xxx",
    "plan_version": "1.0",
    "status": "planned",
    "dry_run_only": true,
    "decision_type": "forward",
    "action_count": 1,
    "approval_required": false,
    "approval_reasons": [],
    "actions": [
      {
        "plan_step": 1,
        "action_id": "forward_J8",
        "action_type": "forward",
        "workflow_node_id": "J8",
        "job_type": "extract_micrographs_multi",
        "execution_mode": "dry_run_only",
        "mcp_tool_name": null,
        "approval_required": false,
        "approval_reasons": [],
        "resolved_parameters": {
          "compute_num_gpus": 4,
          "box_size_pix": 400
        },
        "rollback_target": null,
        "status": "planned"
      }
    ],
    "execution_results": []
  },
  "message": "Dry run only; no CryoSPARC jobs were created or queued."
}
```

## Model Decision JSON

The model output must be a single valid JSON object with no surrounding prose.

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

## V2 Model Input

The model-facing input is now V2 and is built around dataset context plus the
latest completed/failed workflow state:

```json
{
  "schema_version": "2.0",
  "task_type": "workflow_decision",
  "dataset_info": {
    "empiar_id": "EMPIAR-10025",
    "emdb_id": null,
    "resolution": null,
    "input_type": "movies",
    "macromolecules_type": "ribosome",
    "num_of_maps": null,
    "abstract": null,
    "known_workflow_steps": null
  },
  "current_state": {
    "last_node_id": "J33",
    "last_action": "class_2D_new",
    "last_node_status": "completed",
    "last_node_info": {
      "parameters": {
        "class2D_K": 50,
        "compute_num_gpus": 4
      },
      "metrics": {
        "completed": true,
        "failed": false,
        "particles_count": 176623,
        "class_averages_count": 50
      }
    }
  }
}
```

`candidate_actions` are kept inside MCP for validation and execution. They are
not the primary model-facing input. Queue/running states are also kept internal;
the model receives a new V2 payload only after a job completes or fails.

## Verified Real-Job Tests

- `P2/W3/J32`: created from `J7`, ran `extract_micrographs_multi` on
  `g8m192_4090_slurm`, completed with 196 micrographs and 176623 particles.
- `P2/W3/J33`: created from `J8`, ran `class_2D_new` on
  `g8m192_4090_slurm`, completed with 176623 particles and 50 class averages.
- `J33` Slurm submission used `#SBATCH --partition=g8m192` and
  `#SBATCH --gres=gpu:4` on node `4090a`.

## Current Limitations

- The trained model API is not connected yet; current model interaction is
  simulated through JSON payloads and smoke scripts.
- Duplicate-job reuse is not implemented yet, so the next production step is
  to detect equivalent completed/running child jobs before creating new ones.
- Interactive CryoSPARC jobs such as picking inspection and 2D selection still
  require human action in the CryoSPARC UI.
- `known_workflow_steps` can be retrieved from local workflow files, but a
  complete workflow knowledge base has not been built yet.

---

# CryoSPARC Agent Remote 中文说明

当前版本：**V2 Workflow Decision Loop**

`cryosparc_agent_remote` 的目标是把上游模型的结构化决策安全地接入
`172.16.1.2` 上的 CryoSPARC。当前已经支持读取真实 workflow 状态、生成 V2
model 输入、校验/转换模型决策、创建带上游连接的真实 child job、提交到 GPU
lane、内部监控 Job，并在 Job 完成后生成结果上下文。

默认执行路径仍然保守：`execute_model_decision` 和
`execute_v2_model_decision` 会先校验模型输出；在 `dry_run=true` 时只返回执行计划。
在 `dry_run=false` 且动作安全/已审批时，可以创建并提交真实 CryoSPARC Job。

## 架构

系统分为五层：

1. **CryoSPARC 访问层**
   - `cryosparc_client.py` 专门负责创建认证后的 `cryosparc-tools` client。
   - `cryosparc_cli_tools.py` 专门封装 status、version、GPU 查询、worker
     测试等 CLI 命令。

2. **Workflow State 层**
   - 读取 CryoSPARC project/workspace 里的 jobs。
   - 抽象成标准 DAG，包括节点、边、输入、输出、运行中节点和失败节点。
   - 主要文件是 `workflow_state.py`。

3. **Job 说明卡、执行和监控层**
   - `job_specs.py` 保存常见 CryoSPARC job 的“说明卡”：可改参数、是否用
     GPU、是否 interactive、是否需要审批。
   - `job_executor.py` 把校验通过的动作转成通用
     `workspace.create_job(job_type, connections, params)` 执行计划或真实 Job。
   - `job_result.py` 负责内部监控 queue/running，并只在 Job completed/failed 后
     生成给 model 的结果包。

4. **模型决策对齐层**
   - 定义上游模型输出格式。
   - 支持 `forward`、`rollback`、`branch`、`stop`。
   - 根据候选动作校验 action ID、job type、workflow node ID、参数类型和参数范围。
   - `v2_decision_adapter.py` 把 V2 model decision 转成内部 candidate action 执行。
   - 主要文件是 `schemas.py`、`action_registry.py` 和 `v2_decision_adapter.py`。

5. **MCP Tool 层**
   - `model_input_builder.py` 生成 V2 model 输入，包括 `dataset_info` 和
     `current_state`。
   - `known_workflow_retriever.py` 根据本地 workflow 文件检索
     `known_workflow_steps`；匹配不到时填 `null`。
   - 把上述能力暴露成 MCP tools。
   - 主要文件是 `cryosparc_mcp_server.py`。

## 服务器运行命令

在服务器上使用已有的 conda 环境运行：

```bash
cd /ssd1/linweifan/cryosparc_agent
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python cryosparc_mcp_server.py
```

项目使用的 CryoSPARC 路径：

- Master CLI: `/ssd1/linweifan/cryosparc/cryosparc_master/bin/cryosparcm`
- Worker CLI: `/ssd1/linweifan/cryosparc/cryosparc_worker/bin/cryosparcw`
- Config: `/ssd1/linweifan/cryosparc/cryosparc_master/config.sh`

## 已暴露工具

- `get_cryosparc_status`：检查 CryoSPARC master 状态。
- `get_cryosparc_version`：查看 CryoSPARC 版本。
- `get_cryosparc_worker_gpulist`：查看 worker 环境可见 GPU。
- `test_cryosparc_workers`：运行 CryoSPARC worker 验证任务，会创建验证 jobs。
- `create_cryosparc_import_movies_job`：创建真实 Import Movies job。
- `get_workflow_state`：读取 workspace 并返回标准 DAG 快照。
- `get_supported_job_types`：查看当前已有“说明卡”的 job 类型。
- `get_candidate_actions`：根据当前 workflow 状态生成候选动作。
- `validate_model_decision`：只校验模型输出，不执行 Job。
- `execute_model_decision`：校验模型输出并返回执行计划，默认 dry-run。
- `build_model_input_payload` / `get_workflow_decision_context`：生成 V2
  model 输入。
- `validate_v2_model_decision` / `execute_v2_model_decision`：校验并执行 V2
  model decision。

## `execute_model_decision` 当前行为

- 校验失败：返回 `execution_mode="validation_failed"`。
- 校验通过且 `dry_run=true`：返回 `execution_plan`，不创建 Job。
- 当 `compute_num_gpus > 4` 时，`execution_plan` 会标记需要人工审批，
  原因为 `high_gpu_count`。
- `dry_run=false`：对安全/已审批的 forward 动作可以创建并提交真实 Job。

## 当前能完成什么

- 读取 workflow 状态。
- 从真实 workflow 子节点生成候选动作。
- 校验模型输出 JSON。
- 检查参数类型和范围。
- 输出 dry-run 执行计划。
- 创建带上游连接的真实 CryoSPARC child job。
- 提交到默认 GPU lane：`g8m192_4090_slurm`。
- 内部监控 queued/running/completed。
- Job 完成后生成 V2 JSON 给 model。

已完成真实测试：

- `P2/W3/J32`：从 `J7` 创建 `extract_micrographs_multi`，在
  `g8m192_4090_slurm` 完成，输出 176623 particles。
- `P2/W3/J33`：从 `J8` 创建 `class_2D_new`，在 `g8m192_4090_slurm`
  完成，输出 50 个 class averages。

## 还缺什么

- 接入师兄训练的真实 model API。
- 防重复提交机制：已有同类型/同参数/同上游 Job 时应复用，不再新建。
- interactive jobs 的人工确认流程。
- 完整的 known workflow 知识库。
