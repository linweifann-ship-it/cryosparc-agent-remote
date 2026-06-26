# CryoSPARC Agent Project

Current version: **Dry-Run Decision Executor v0.2**

The project code is in [`cryosparc_agent_remote/`](cryosparc_agent_remote/).
It implements MCP tools that connect an upstream decision model to CryoSPARC in
a safe dry-run-first workflow.

Main capabilities:

- read CryoSPARC workflow state as a normalized DAG
- generate candidate actions from the current workflow node
- validate model decision JSON
- return dry-run execution plans through `execute_model_decision`
- use job metadata cards plus a generic executor instead of one wrapper per job
- keep live execution disabled until approval policy and job wrappers are added

See [`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md) for
the full bilingual documentation.

---

# CryoSPARC Agent 项目

当前版本：**Dry-Run Decision Executor v0.2**

项目代码位于 [`cryosparc_agent_remote/`](cryosparc_agent_remote/)。它实现了一组
MCP tools，用于把上游模型决策安全接入 CryoSPARC，并默认采用 dry-run 优先的
执行流程。

主要能力：

- 把 CryoSPARC workflow 读取成标准 DAG
- 根据当前 workflow 节点生成候选动作
- 校验模型输出 JSON
- 通过 `execute_model_decision` 返回 dry-run 执行计划
- 用 job 说明卡和通用执行器替代“每个 job 一个 wrapper”
- 在接入审批策略和真实 Job Wrapper 前保持真实执行关闭

完整中英文说明见
[`cryosparc_agent_remote/README.md`](cryosparc_agent_remote/README.md)。
