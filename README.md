# CryoSPARC Agent Project

Current version: **Workflow Decision Executor v0.3**

The project code is in [`cryosparc_agent_remote/`](cryosparc_agent_remote/).
It implements MCP tools that connect an upstream decision model to CryoSPARC in
a safe dry-run-first workflow.

Main capabilities:

- read CryoSPARC workflow state as a normalized DAG
- generate candidate actions from the current workflow node
- validate model decision JSON
- build dry-run execution plans through `execute_model_decision`
- execute supported CryoSPARC jobs behind explicit approval gates
- build V2 model inputs, adapt model decisions, and package terminal job results
- reject decisions generated from stale workflow snapshots or candidate sets

See [`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md) for
the full bilingual documentation.

## Deep Agents harness

The `deepagents` branch adds a single-main-agent replacement for the custom
autonomous closed-loop driver. It keeps the existing MCP server and all
workflow/scientific logic intact; Deep Agents only drives the loop through
those tools.

```bash
cd cryosparc-agent-remote
/opt/anaconda3/bin/python -m pip install -r requirements-deepagents.txt

# Safe smoke: starts only the MCP stdio server and lists tools. No model call,
# workflow read, validation, job creation, or CryoSPARC execution.
/opt/anaconda3/bin/python scripts/run_deepagents_closed_loop.py \
  --project P2 --workspace W1 \
  --server-python /opt/anaconda3/bin/python \
  --project-dir "$PWD/cryosparc_agent_remote" --smoke
```

For a dry-run decision round, set the existing OpenAI-compatible configuration
through flags or environment (the harness has no key/base-URL/model default):

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=...
/opt/anaconda3/bin/python scripts/run_deepagents_closed_loop.py \
  --project P2 --workspace W1 \
  --server-python /path/to/server-python \
  --project-dir /path/to/cryosparc_agent_remote
```

Without `--execute`, `execute_v2_model_decision` is instructed to remain
`dry_run=true`. `--execute` is an explicit live-workflow opt-in.

---

# CryoSPARC Agent 项目

当前版本：**Workflow Decision Executor v0.3**

项目代码位于 [`cryosparc_agent_remote/`](cryosparc_agent_remote/)。它实现了一组
MCP tools，用于把上游模型决策安全接入 CryoSPARC，并默认采用 dry-run 优先的
执行流程。

主要能力：

- 把 CryoSPARC workflow 读取成标准 DAG
- 根据当前 workflow 节点生成候选动作
- 校验模型输出 JSON
- 通过 `execute_model_decision` 构建 dry-run 执行计划
- 在明确审批门控下执行已支持的 CryoSPARC Job
- 构建 V2 模型输入、适配模型决策并封装终态 Job 结果
- 拒绝基于过期 workflow 快照或候选集生成的决策

完整中英文说明见
[`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md)。
