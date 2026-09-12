---
title: "CryoAgent: An Automated Electron Microscopy Data Processing Assistant"
subtitle: "First Project Progress Report"
author: "Lisongyang"
date: "July 2026"
---

# CryoAgent: An Automated Electron Microscopy Data Processing Assistant

First Project Progress Report

![](ppt_assets_group_meeting_20260706/cryoagent_title_visual.svg){width=85%}

# Why Workflow Automation Matters

- Modern cryo-EM can acquire large datasets quickly, but downstream data processing remains a major bottleneck.
- The pipeline is multi-step and iterative, with noisy data, unknown poses, structural heterogeneity, and expensive computation.
- Important subtasks and workflow choices still often require substantial expert effort.

**Takeaway:** modern software automates many steps, but end-to-end workflow control is still far from push-button.

# What We Want the Agent to Do

- Input: dataset context plus current workflow state
- Output: next data-processing action and parameters in structured JSON
- Target: support a closed-loop execution pipeline rather than isolated prediction

![](ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg){width=82%}

# Why This Is Hard

- Decisions are sequential and state-dependent.
- The model must use both scientific context and runtime evidence.
- The same system should support known workflow reuse, exploratory planning on unseen datasets, and human-interactive processing steps.

**Challenge:** the correct action depends on what has already happened, what outputs are available, and what execution constraints apply next.

# Related Work and the Missing Layer

- cryo-EM platforms such as cryoSPARC, RELION, and Warp automate important processing modules and preprocessing steps.
- Deep learning methods such as Topaz, cryoDRGN, and e2gmm improve particle picking and heterogeneity analysis.
- Recent agentic AI research suggests that scientific workflows can be assisted by planning-and-tool-using models.
- Gap: these directions are still disconnected, and there is no unified agent layer for dataset-aware, state-aware, executable EM workflow decision making.

![](ppt_assets_group_meeting_20260706/cryoagent_research_gap_map.svg){width=88%}

# System-Level Idea

- Separate decision generation from execution control.
- Model: predicts the next workflow action from V2 context.
- MCP server: builds state, validates output, plans execution, runs jobs, and monitors status.
- JSON is the stable contract between the two layers.

![](ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg){width=76%}

# Outer MCP Server Architecture

- Implemented repository: `/ssd1/linweifan/cryosparc_agent`
- Five implemented layers:
- CryoSPARC access
- workflow state extraction
- job metadata and execution
- decision alignment
- model input and MCP tools

![](ppt_assets_group_meeting_20260706/cryosparc_mcp_architecture_preview.png){width=88%}

# From State to Decision to Action

1. MCP collects dataset metadata and workflow state.
2. Model predicts `forward`, `branch`, `rollback`, or `stop`.
3. MCP adapts and validates the decision.
4. MCP creates a plan, executes when allowed, and monitors the result.
5. The result becomes the next round's input.

![](ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg){width=74%}

# Unified V2 Input Schema

- Data sources: EMDB XML, workflow JSON, runtime logs, dataset labels, and live workspace state
- Top-level structure:
- `dataset_info`
- `current_state`
- Supports both known-workflow and no-workflow scenarios

**Design choice:** use one interface for both workflow reuse and exploratory planning.

# Model Adaptation Plan

- Base model: `Qwen3.6-27B`
- SFT teaches workflow decision format and behavior.
- CPT injects broader cryo-EM domain knowledge from literature.
- Main comparison:
- `Base -> SFT`
- `Base -> CPT -> SFT`

**Hypothesis:** CPT should improve no-workflow generalization and state-conditioned decision quality.

# Current Progress: Result Snapshot

- `No-workflow validation (137 samples)`: `99.27%` valid JSON, `96.35%` decision-type accuracy, `39.42%` exact action+parameter match.
- `Known-workflow upper bound (152 samples)`: `98.68%` exact match in a strong-context setting.
- `MCP live validation`: real cryoSPARC jobs `J36`, `J42`, and `J59` were created and monitored successfully.
- `Human-in-the-loop handling`: interactive job `J37` was correctly paused with `ready_for_model = false`.

![](ppt_assets_group_meeting_20260706/cryosparc_mcp_architecture_简版_preview.png){width=82%}

# What Is Distinct About This Project

- Workflow-level decision control rather than isolated cryo-EM subtasks
- Unified schema for known-workflow reuse and no-workflow exploration
- Runtime-aware state construction from logs and workflow records
- MCP server as execution, monitoring, validation, and safety layer
- `CPT -> SFT` as a domain-aware adaptation path

![](ppt_assets_group_meeting_20260706/cryoagent_research_gap_map.svg){width=82%}

# What Still Needs Work

- Exact node-level and branch decisions remain harder than coarse action selection.
- Parameter prediction still needs stronger constraints and validation.
- Interactive cryoSPARC steps require a cleaner human-in-the-loop protocol.
- The MCP execution policy should evolve from smoke-test rules to production governance.

**Current view:** the direction looks viable, but fine-grained reliability is still the main challenge.

# Next Steps

- Run the first CPT experiment and compare against the current SFT baseline
- Expand MCP integration from smoke tests to stable closed-loop runs on representative datasets
- Evaluate both decision quality and execution success
- Refine the human-in-the-loop boundary for interactive steps

**Discussion:** is the model/MCP boundary and human-in-the-loop strategy reasonable?
