# CryoSPARC Agent Remote

Current version: **Dry-Run Decision Executor v0.2**

`cryosparc_agent_remote` provides MCP tools for safely connecting an upstream
decision model to a CryoSPARC instance on `172.16.1.2`. The project currently
supports workflow-state extraction, candidate action generation, model decision
validation, and dry-run execution planning.

The default execution path is intentionally safe: `execute_model_decision`
validates the model output and returns a planned execution, but it does **not**
create or queue CryoSPARC jobs while `dry_run=true`.

## Architecture

The system has four layers:

1. **CryoSPARC access layer**
   - Wraps CryoSPARC CLI commands and `cryosparc-tools`.
   - Reads credentials from environment variables or CryoSPARC `config.sh`.
   - Lives mainly in `cryosparc_cli_tools.py`.

2. **Workflow state layer**
   - Reads jobs from a CryoSPARC project/workspace.
   - Converts the workspace into a normalized DAG with nodes, edges, inputs,
     outputs, running nodes, failed nodes, and a stable `state_snapshot_id`.
   - Lives in `workflow_state.py`.

3. **Decision alignment layer**
   - Defines the upstream model output schema.
   - Supports `forward`, `rollback`, `branch`, and `stop`.
   - Validates action IDs, job types, workflow node IDs, parameter types,
     parameter ranges, `state_snapshot_id`, and `candidate_set_id`.
   - Lives in `schemas.py` and `action_registry.py`.

4. **MCP tool layer**
   - Exposes the project as MCP tools.
   - Lives in `cryosparc_mcp_server.py`.

## Server Run Command

Use the server-side conda environment that already has `mcp` and
`cryosparc-tools` installed:

```bash
cd /ssd1/linweifan/cryosparc_agent
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python cryosparc_mcp_server.py
```

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

- `state_snapshot_id`
- `nodes`
- `edges`
- `root_nodes`
- `terminal_nodes`
- `running_nodes`
- `failed_nodes`
- `node_mapping`

The generated timestamp is not included in the snapshot hash, so unchanged
workspaces produce stable `state_snapshot_id` values.

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

All generated actions currently remain `dry_run_only`.

### `validate_model_decision`

Validates an upstream model decision JSON against schema version `1.0` and a
candidate action set.

The validator checks:

- top-level schema fields and value ranges
- `decision_type` rules
- selected `action_id` membership
- `action_type`, `workflow_node_id`, and `job_type` consistency
- allowed parameters and parameter type/range constraints
- `state_snapshot_id` freshness
- `candidate_set_id` freshness

This tool is validation-only. It does not create or enqueue CryoSPARC jobs.

### `execute_model_decision`

Validates a model decision and returns an execution plan.

Inputs:

- `decision`: upstream model decision JSON.
- `project_uid`, `workspace_uid`, `current_node_id`: optional live context used
  to regenerate candidate actions and freshness IDs.
- `candidate_actions`: optional caller-supplied candidate action list.
- `dry_run`: optional boolean. Default: `true`.

Behavior:

- If validation fails, returns `execution_mode="validation_failed"`.
- If validation succeeds and `dry_run=true`, returns an `execution_plan` and
  compatibility field `planned_actions`; it does not create jobs.
- If `dry_run=false`, returns `live_execution_not_implemented` until human
  approval policy and job wrappers are added.

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
    "state_snapshot_id": "state_xxx",
    "candidate_set_id": "candidates_xxx",
    "action_count": 1,
    "approval_required": false,
    "approval_reasons": [],
    "actions": [
      {
        "plan_step": 1,
        "action_id": "forward_J8",
        "action_type": "forward",
        "workflow_node_id": "node_extract_micrographs_multi_001",
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
  "planned_actions": [
    "same as execution_plan.actions"
  ],
  "message": "Dry run only; no CryoSPARC jobs were created or queued."
}
```

## Model Decision JSON

The model output must be a single valid JSON object with no surrounding prose.

```json
{
  "schema_version": "1.0",
  "state_snapshot_id": "state_xxx",
  "candidate_set_id": "candidates_xxx",
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

## Current Limitations

- Live execution is not implemented yet.
- Human approval policy is not implemented yet.
- Only Import Movies has a direct real-job creation wrapper.
- Additional wrappers are still needed for extraction, 2D classification,
  refinement, job queueing, and job status/result tracking.

---

# CryoSPARC Agent Remote 中文说明

当前版本：**Dry-Run Decision Executor v0.2**

`cryosparc_agent_remote` 的目标是把上游模型的结构化决策安全地接入
`172.16.1.2` 上的 CryoSPARC。当前已经支持读取 workflow 状态、生成候选动作、
校验模型决策，以及 dry-run 执行计划。

默认执行路径是安全的：`execute_model_decision` 会先校验模型输出，然后只返回
“计划执行什么”，在 `dry_run=true` 时不会创建或排队任何 CryoSPARC Job。

## 架构

系统分为四层：

1. **CryoSPARC 访问层**
   - 封装 CryoSPARC CLI 和 `cryosparc-tools`。
   - 从环境变量或 CryoSPARC `config.sh` 读取认证信息。
   - 主要文件是 `cryosparc_cli_tools.py`。

2. **Workflow State 层**
   - 读取 CryoSPARC project/workspace 里的 jobs。
   - 抽象成标准 DAG，包括节点、边、输入、输出、运行中节点、失败节点和稳定的
     `state_snapshot_id`。
   - 主要文件是 `workflow_state.py`。

3. **模型决策对齐层**
   - 定义上游模型输出格式。
   - 支持 `forward`、`rollback`、`branch`、`stop`。
   - 校验 action ID、job type、workflow node ID、参数类型、参数范围、
     `state_snapshot_id` 和 `candidate_set_id`。
   - 主要文件是 `schemas.py` 和 `action_registry.py`。

4. **MCP Tool 层**
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
- `get_candidate_actions`：根据当前 workflow 状态生成候选动作。
- `validate_model_decision`：只校验模型输出，不执行 Job。
- `execute_model_decision`：校验模型输出并返回执行计划，默认 dry-run。

## `execute_model_decision` 当前行为

- 校验失败：返回 `execution_mode="validation_failed"`。
- 校验通过且 `dry_run=true`：返回 `execution_plan` 和兼容字段
  `planned_actions`，不创建 Job。
- `dry_run=false`：返回 `live_execution_not_implemented`，因为真实执行还需要先接入
  Human Approval 和更多 Job Wrapper。

## 当前能完成什么

- 读取 workflow 状态。
- 从真实 workflow 子节点生成候选动作。
- 校验模型输出 JSON。
- 检查参数类型和范围。
- 检查 state/candidate ID 是否过期。
- 输出 dry-run 执行计划。

## 还缺什么

- 真实执行器。
- Human Approval 审批策略。
- 更多 CryoSPARC Job Wrapper。
- Job queue/status/result 查询。
- 与上游模型的端到端联调。
