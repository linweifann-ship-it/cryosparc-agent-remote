# CryoSPARC Agent Remote

Current version: **V2 Workflow Decision Loop**

`cryosparc_agent_remote` provides MCP tools for safely connecting an upstream
decision model to a CryoSPARC instance on `172.16.1.2`. The project currently
supports real workflow-state extraction, V2 model input generation, model
decision validation/adaptation, real child-job creation, GPU-lane submission,
internal job monitoring, completed-job result packaging, XML dataset metadata
loading, known-workflow retrieval, and generic CryoSPARC job planning.

The default execution path is still conservative: `execute_model_decision` and
`execute_v2_model_decision` validate model output first and return a planned
execution while `dry_run=true`. Live execution is available for approved
forward actions and has been tested on the `g8m192_4090_slurm` lane.

V2 model input no longer exposes `candidate_actions` to the model. The model
can propose a CryoSPARC `action`/`job_type` directly. MCP uses candidates only
as an internal helper when they are available; if no candidate matches, MCP can
still build a generic `workspace.create_job(job_type, connections, params)`
plan from the model decision.

## Architecture

The current architecture separates "what the model sees" from "what MCP can
execute". The model sees only V2 workflow context; MCP owns state management,
connection resolution, validation, queueing, monitoring, and the final
CryoSPARC API calls.

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
     It prefers model-supplied `connections`, then falls back to internally
     inferred inputs from workflow context.
   - Non-interactive CPU/import jobs are queued with `job.queue()` even when no
     GPU lane is set. GPU jobs use the configured default lane,
     currently `g8m192_4090_slurm`.
   - `job_result.py` keeps queue/running states internal and packages only
     completed/failed results for model-facing updates.

4. **Decision alignment layer**
   - Defines the upstream model output schema.
   - Supports `forward`, `rollback`, `branch`, and `stop`.
   - Adapts compact V2 decisions into executable plans.
   - Candidate actions are optional internal hints, not hard requirements.
     Unknown future job types are planned as generic CryoSPARC jobs, with
     warnings and approval policy preserved.
   - This means MCP no longer needs to pre-enumerate every possible CryoSPARC
     job type before the model can request it. For unfamiliar jobs, the model
     should provide the CryoSPARC `job_type`, valid `parameters`, and, when MCP
     cannot infer them, explicit input `connections`.
   - Lives in `schemas.py`, `action_registry.py`, and
     `v2_decision_adapter.py`.

5. **Model input and MCP tool layer**
   - `model_input_builder.py` builds V2 model-facing payloads with
     `dataset_info` and `current_state`, including `recent_nodes` so the model
     can see short workflow history.
   - `dataset_xml.py` extracts EMDB/XML metadata into `dataset_info`, including
     pixel size, accelerating voltage, spherical aberration, total exposure
     dose, and symmetry.
   - `known_workflow_retriever.py` fills `known_workflow_steps` from local
     workflow files when a dataset match is available, including CryoSPARC
     workflow JSON files whose `jobs` are stored as a dictionary; otherwise it
     returns `null`.
   - `scripts/extract_dataset_info_xml.py` turns an EMDB XML file into a
     reusable dataset JSON file for closed-loop tests.
   - Exposes the project as MCP tools.
   - Lives in `cryosparc_mcp_server.py`.

The model-facing V2 payload intentionally omits `candidate_actions`. A typical
payload is:

```json
{
  "schema_version": "2.0",
  "task_type": "workflow_decision",
  "dataset_info": {
    "emdb_id": "EMD-6287",
    "empiar_id": "EMPIAR-10025",
    "pixel_size_A": 0.982,
    "accelerating_voltage_kv": 300,
    "spherical_aberration_mm": 2.7,
    "total_exposure_dose_e_per_A2": 53,
    "symmetry": "D7",
    "known_workflow_steps": []
  },
  "current_state": {
    "last_node_id": "J42",
    "last_action": "import_micrographs",
    "last_node_status": "completed",
    "recent_nodes": [
      {"node_id": "J42", "job_type": "import_micrographs", "status": "completed"}
    ],
    "last_node_info": {}
  }
}
```

The model can answer with a compact decision:

```json
{
  "schema_version": "1.0",
  "decision_type": "forward",
  "action": "patch_ctf_estimation_multi",
  "parameters": {"compute_num_gpus": 1},
  "connections": {
    "exposures": {
      "source_job_uid": "J42",
      "source_output": "imported_micrographs"
    }
  },
  "reason": "Imported micrographs are ready for CTF estimation.",
  "confidence": 0.9,
  "risk_flags": [],
  "evidence": []
}
```

MCP then builds a dry-run plan or live job. Queue/running states stay internal
to MCP. Only completed/failed jobs produce the next model-facing V2 payload.

## Current Verified Workflow

The latest real server test used project `P2`, workspace `W5`
(`W5 Agent_xml_abi`), and EMDB XML metadata:

```bash
/hdd1/huangjianhua/agent/data/experiment/emd-6287.xml
```

The XML parser extracted:

- Pixel size: `0.982 A`
- Accelerating voltage: `300 kV`
- Spherical aberration: `2.7 mm`
- Total exposure dose: `53 e/A^2`
- Symmetry: `D7`

The known workflow file was:

```bash
/ssd1/linweifan/cryosparc_agent/reports/model_closed_loop/empiar-10025-workflow-standard.json
```

It contains `homo_abinit` and does not require `import_volumes`. The server data
directory contains averaged micrographs:

```bash
/home/share/empiar/10025/data/14sep05c_averaged_196/*.mrc
```

Because those files are micrographs rather than movies, the live run used
`import_micrographs` as the first job:

- `J42 import_micrographs`: completed, `196` imported micrographs, `0` failed.
- `J43 patch_ctf_estimation_multi`: submitted from `J42.imported_micrographs`
  to `exposures`, running during the last check with `100` completed exposures
  and `96` incomplete exposures.

One earlier test job, `J41`, was created before the import-queue fix and stayed
in `building`; it should be ignored in decision context or cleaned up manually
before formal benchmark runs.

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
local Qwen model with `enable_thinking=False`, parses the returned JSON, adapts
it to an execution plan, and returns that planned execution. Candidate actions
are optional internal helpers; they are not required for generic job planning.
The script does not create or queue a CryoSPARC job unless `--live` is added.

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

Generated actions may be `dry_run_only` when they reproduce an existing
reference child job, or `create_job` when MCP can create a new CryoSPARC job.

GPU actions with `compute_num_gpus > 4` are marked for human approval in the
generated `execution_plan` with approval reason `high_gpu_count`.

### `get_supported_job_types`

Returns the job types that currently have explicit local metadata in
`job_specs.py`. Unknown job types can still be represented, but they default to
human approval before live execution.

### `validate_model_decision`

Validates an upstream model decision JSON against schema version `1.0`.
Candidate actions may be supplied, but they are optional.

The validator checks:

- top-level schema fields and value ranges
- `decision_type` rules
- `action_id`, `workflow_node_id`, and `job_type` consistency when a candidate
  is matched
- known parameter type/range constraints when local job metadata exists
- missing required parameters for known job types

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
- If no internal candidate matches, MCP can still create a generic plan from
  the model's `job_type`, `parameters`, and optional `connections`.

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
The compact V2 adapter accepts either `action`/`job_type` or explicit
`selected_actions`. For unfamiliar jobs, include `connections` when MCP cannot
infer CryoSPARC inputs from workflow context.

Compact example:

```json
{
  "schema_version": "1.0",
  "decision_type": "forward",
  "action": "homo_abinit",
  "parameters": {},
  "connections": {
    "particles": {
      "source_job_uid": "J13",
      "source_output": "particles_selected"
    }
  },
  "reason": "Selected particles are ready for ab initio reconstruction.",
  "confidence": 0.9,
  "risk_flags": [],
  "evidence": []
}
```

Internal expanded shape:

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
    "emdb_id": "EMD-6287",
    "resolution": null,
    "input_type": "movies",
    "macromolecules_type": "20S proteasome",
    "num_of_maps": null,
    "abstract": "2.8 Angstrom resolution reconstruction of the T20S proteasome",
    "known_workflow_steps": null,
    "pixel_size_A": 0.982,
    "accelerating_voltage_kv": 300,
    "spherical_aberration_mm": 2.7,
    "total_exposure_dose_e_per_A2": 53,
    "symmetry": "D7"
  },
  "current_state": {
    "last_node_id": "J33",
    "last_action": "class_2D_new",
    "last_node_status": "completed",
    "recent_nodes": [
      {"node_id": "J32", "job_type": "extract_micrographs_multi", "status": "completed"},
      {"node_id": "J33", "job_type": "class_2D_new", "status": "completed"}
    ],
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

`candidate_actions` are not model-facing. MCP may still compute them internally
to infer common connections, but lack of a candidate no longer blocks generic
job planning. Queue/running states are also kept internal; the model receives a
new V2 payload only after a job completes or fails.

XML user input can be converted to `dataset_info` with:

```bash
env PYTHONPATH=/ssd1/linweifan/cryosparc_agent \
  /ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python \
  scripts/extract_dataset_info_xml.py \
  --xml-file /hdd1/huangjianhua/agent/data/experiment/emd-6287.xml \
  --output-json-file reports/model_closed_loop/dataset_info_emd_6287.json
```

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
lane、内部监控 Job、XML 数据集信息解析、known workflow 检索，并在 Job
完成后生成结果上下文。

默认执行路径仍然保守：`execute_model_decision` 和
`execute_v2_model_decision` 会先校验模型输出；在 `dry_run=true` 时只返回执行计划。
在 `dry_run=false` 且动作安全/已审批时，可以创建并提交真实 CryoSPARC Job。

V2 输入不再把 `candidate_actions` 发给 model。model 可以直接返回
`action`/`job_type`；MCP 会优先用内部 candidate 辅助推断连接，如果没有匹配
candidate，也可以根据 model 给出的 `job_type`、`parameters` 和可选
`connections` 生成通用 CryoSPARC Job 计划。

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
     连接优先使用 model 明确给出的 `connections`；没有时再尝试用 MCP 内部
     workflow context 推断。
   - `job_result.py` 负责内部监控 queue/running，并只在 Job completed/failed 后
     生成给 model 的结果包。

4. **模型决策对齐层**
   - 定义上游模型输出格式。
   - 支持 `forward`、`rollback`、`branch`、`stop`。
   - `candidate_actions` 只是内部辅助，不再是硬限制。
   - 未知的新 CryoSPARC job type 可以被规划成 generic job，并保留 warning 和
     审批策略。
   - `v2_decision_adapter.py` 把 V2 model decision 转成可执行计划。
   - 主要文件是 `schemas.py`、`action_registry.py` 和 `v2_decision_adapter.py`。

5. **MCP Tool 层**
   - `model_input_builder.py` 生成 V2 model 输入，包括 `dataset_info` 和
     `current_state`，并通过 `recent_nodes` 给 model 提供最近 workflow 历史。
   - `dataset_xml.py` 从 EMDB XML 中提取用户输入的数据集信息，包括 pixel
     size、加速电压、球差、总剂量和对称性。
   - `known_workflow_retriever.py` 根据本地 workflow 文件检索
     `known_workflow_steps`；支持 CryoSPARC workflow JSON 的 `jobs` 字典格式；
     匹配不到时填 `null`。
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
- `get_candidate_actions`：根据当前 workflow 状态生成内部候选动作，主要用于
  MCP 推断连接和诊断，不再发给 model 作为主输入。
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
- 没有匹配 candidate 时，不再直接拒绝；MCP 会按 `job_type`、`parameters`
  和可选 `connections` 生成 generic job plan。

## 当前能完成什么

- 读取 workflow 状态。
- 从真实 workflow 子节点生成内部候选动作，用于辅助推断连接。
- 校验模型输出 JSON。
- 检查参数类型和范围。
- 输出 dry-run 执行计划。
- 创建带上游连接的真实 CryoSPARC child job。
- 对未登记的新 job type 生成 generic job plan。
- 从 EMDB XML 生成 `dataset_info`。
- 从本地标准 workflow JSON 生成 `known_workflow_steps`，包括 ab initio 路线。
- 提交到默认 GPU lane：`g8m192_4090_slurm`。
- 内部监控 queued/running/completed。
- Job 完成后生成 V2 JSON 给 model。

已完成真实测试：

- `P2/W3/J32`：从 `J7` 创建 `extract_micrographs_multi`，在
  `g8m192_4090_slurm` 完成，输出 176623 particles。
- `P2/W3/J33`：从 `J8` 创建 `class_2D_new`，在 `g8m192_4090_slurm`
  完成，输出 50 个 class averages。

## 还缺什么

- 接入师兄训练的真实 model API；当前已能在服务器直接调用本地模型做闭环测试。
- `import_movies` 真实 live 创建还需要用户提供 movie 路径 `blob_paths` 和必要的
  gain reference 路径。
- 防重复提交机制：已有同类型/同参数/同上游 Job 时应复用，不再新建。
- interactive jobs 的人工确认流程。
- 完整的 known workflow 知识库。
