# CryoSPARC Agent Project

Current version: **V2 Workflow Decision Loop**

This repository contains the local project used to connect an upstream decision
model to CryoSPARC through MCP tools. The active code lives in
[`cryosparc_agent_remote/`](cryosparc_agent_remote/).

The current system is no longer dry-run only. It can read real CryoSPARC
workflow state, build V2 model-facing JSON, validate/adapt model decisions,
create connected child jobs, submit approved GPU jobs, monitor jobs internally,
and return a new V2 payload only after a job completes or fails.

## Repository Layout

- `cryosparc_agent_remote/`: MCP server, CryoSPARC client, workflow parser,
  job executor, V2 model input builder, tests, and smoke scripts.
- `MCP_SERVER_INPUT_FORMAT.md`: external V2 JSON format reference from the
  model side.
- `architecture_figure/`: project architecture diagrams.
- `cryo_em_agent_research/`: background research materials and literature notes.

Only `cryosparc_agent_remote/` is required to run the current MCP agent code.

## Current Capabilities

- Read real CryoSPARC project/workspace workflow state as a normalized DAG.
- Generate internal candidate actions from real upstream/downstream job links.
- Build V2 model input with `dataset_info` and `current_state`.
- Keep `candidate_actions` internal for validation and execution.
- Validate model decision JSON.
- Adapt compact V2 model decisions into internal executable actions.
- Create CryoSPARC child jobs with real upstream input connections.
- Submit approved GPU jobs to the default lane `g8m192_4090_slurm`.
- Keep queue/running states inside MCP; do not send them to the model.
- Generate a new V2 payload for the model after a job completes or fails.

## Verified Real Tests

Test project/workspace: `P2 / W3`

- `J32`: created from `J7`, ran `extract_micrographs_multi` on
  `g8m192_4090_slurm`, completed with 196 micrographs and 176623 particles.
- `J33`: created from `J8`, ran `class_2D_new` on `g8m192_4090_slurm`,
  completed with 176623 particles and 50 class averages.
- `J33` Slurm submission used `#SBATCH --partition=g8m192` and
  `#SBATCH --gres=gpu:4` on node `4090a`.

## Local And Server Paths

- Local project: `/Users/Zhuanz/Documents/杭州服务器/cryosparc_agent_remote`
- Server project: `/ssd1/linweifan/cryosparc_agent`
- GitHub remote:
  `git@github-shipit:linweifann-ship-it/cryosparc-agent-remote.git`

Server run command:

```bash
cd /ssd1/linweifan/cryosparc_agent
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python cryosparc_mcp_server.py
```

## Current Limitations

- The trained model API is not connected yet.
- Duplicate-job reuse is not implemented yet.
- Interactive CryoSPARC jobs still require human action in the CryoSPARC UI.
- `known_workflow_steps` can be loaded from local workflow files, but a complete
  workflow knowledge base has not been built yet.

See [`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md) for
the detailed bilingual module documentation.

---

# CryoSPARC Agent 项目

当前版本：**V2 Workflow Decision Loop**

这个仓库用于把上游决策模型通过 MCP tools 接入 CryoSPARC。当前主要代码位于
[`cryosparc_agent_remote/`](cryosparc_agent_remote/)。

现在系统已经不是只做 dry-run。它可以读取真实 CryoSPARC workflow，生成给
model 的 V2 JSON，校验/转换 model decision，创建带上游连接的 child job，
提交安全的 GPU Job，内部监控 Job，并且只在 Job completed/failed 后重新生成
V2 JSON 发给 model。

## 目录结构

- `cryosparc_agent_remote/`：MCP server、CryoSPARC client、workflow 解析、
  job executor、V2 model input builder、测试和 smoke scripts。
- `MCP_SERVER_INPUT_FORMAT.md`：模型侧给出的 V2 JSON 格式参考。
- `architecture_figure/`：项目架构图。
- `cryo_em_agent_research/`：背景调研和文献资料。

当前运行 MCP agent 只需要 `cryosparc_agent_remote/`。

## 当前能力

- 读取真实 CryoSPARC project/workspace workflow。
- 根据真实上下游 Job 关系生成 MCP 内部 candidate actions。
- 生成 V2 model 输入：`dataset_info` + `current_state`。
- `candidate_actions` 只作为 MCP 内部校验和执行结构，不作为 model 主输入。
- 校验 model decision JSON。
- 把 V2 model decision 转成内部可执行动作。
- 创建带真实上游输入连接的 CryoSPARC child job。
- 提交已审批/安全的 GPU Job 到默认 lane：`g8m192_4090_slurm`。
- queue/running 阶段只由 MCP 内部保存和监控，不发给 model。
- Job completed/failed 后再生成新的 V2 JSON 给 model。

## 已完成真实测试

测试项目/workspace：`P2 / W3`

- `J32`：从 `J7` 创建 `extract_micrographs_multi`，在
  `g8m192_4090_slurm` 完成，输出 196 个 micrographs 和 176623 个 particles。
- `J33`：从 `J8` 创建 `class_2D_new`，在 `g8m192_4090_slurm` 完成，
  输出 176623 个 particles 和 50 个 class averages。
- `J33` 的 Slurm 提交确认使用 `#SBATCH --partition=g8m192` 和
  `#SBATCH --gres=gpu:4`，运行节点是 `4090a`。

## 本地和服务器位置

- 本地项目：`/Users/Zhuanz/Documents/杭州服务器/cryosparc_agent_remote`
- 服务器项目：`/ssd1/linweifan/cryosparc_agent`
- GitHub remote:
  `git@github-shipit:linweifann-ship-it/cryosparc-agent-remote.git`

服务器运行命令：

```bash
cd /ssd1/linweifan/cryosparc_agent
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python cryosparc_mcp_server.py
```

## 当前还缺什么

- 接入师兄训练的真实 model API。
- 防重复提交机制：已有同类型/同参数/同上游 Job 时应复用，不再新建。
- interactive CryoSPARC Job 仍需要人工在 UI 中处理。
- `known_workflow_steps` 可以从本地 workflow 文件读取，但还没有完整 workflow
  知识库。

详细模块说明见
[`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md)。
