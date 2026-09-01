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
