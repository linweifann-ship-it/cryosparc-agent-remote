# CryoAgent Group Meeting PPT Content

This document expands the current outline into slide-ready English content. It is designed for a first project report, so the emphasis is on problem framing, overall system design, current progress, and next-step direction rather than dense technical detail.

## Slide 1. Title

### Title
CryoAgent: An Automated Electron Microscopy Data Processing Assistant

### Subtitle
First Project Progress Report

### On-slide text
- Goal: move from expert-driven workflow operation toward model-guided electron microscopy data-processing automation
- Scope of this report: problem framing, system design, current prototype status, and next steps

### Speaker note
This talk introduces CryoAgent as a system project rather than only a model project. The long-term goal is a general assistant for electron microscopy data processing, while the current validation platform is cryoSPARC. The central idea is to let a domain-adapted model make the next workflow decision, while an MCP server handles state construction, validation, execution, and monitoring around it.

### Suggested image
- Use an original title visual with a cryo-EM style background plus a workflow loop overlay.
- Current recommendation: use `ppt_assets_group_meeting_20260706/cryoagent_title_visual.svg`.

## Slide 2. Background and Motivation

### Slide title
Why Workflow Automation Matters

### On-slide text
- Modern cryo-EM can acquire large datasets quickly, but downstream data processing remains a major bottleneck
- The pipeline is multi-step and iterative, with noisy data, unknown poses, structural heterogeneity, and expensive computation
- Important subtasks and workflow choices still often require substantial expert effort

### Speaker note
This slide is directly supported by the literature. Punjani et al. describe data processing as a severe bottleneck even when data acquisition is fast. Singer and Sigworth review the computational difficulty of single-particle cryo-EM in terms of noise, unknown pose estimation, and heterogeneity. Bepler et al. further show that some subtasks, such as particle identification, can still require months of manual effort. Together, these results justify why a workflow-level assistant is still needed.

### Suggested image
- A simple workflow pipeline graphic or a cryo-EM micrograph/molecule composition image.
- If you want a real scientific background image, use a self-made screenshot from EMPIAR/EMDB landing pages or a public cryo-EM overview figure with citation.

## Slide 3. Problem Statement

### Slide title
What We Want the Agent to Do

### On-slide text
- Input: dataset context plus current workflow state
- Output: next data-processing action and parameters in structured JSON
- Target: support a closed-loop execution pipeline rather than isolated prediction

### Speaker note
Our goal is not just to classify one step or answer a question. We want a decision module that can be inserted into a real processing loop and repeatedly decide the next action as the workflow evolves. cryoSPARC is the current execution platform, but the system framing is intentionally broader.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg`.

## Slide 4. Why This Problem Is Challenging

### Slide title
Why This Is Hard

### On-slide text
- Decisions are sequential and state-dependent
- The model must use both scientific context and runtime evidence
- The same system should support:
- known workflow reuse
- exploratory planning on unseen datasets
- human-interactive processing steps

### Speaker note
This is harder than ordinary task prediction because the correct action depends on what has already happened, what outputs are available, whether the last job succeeded, and whether the next step needs human supervision.

### Suggested image
- Use a layered challenge diagram or a three-column visual showing dataset context, runtime state, and execution constraints.
- Current recommendation: use `ppt_assets_group_meeting_20260706/cryoagent_research_gap_map.svg` or build a simple text-visual slide.

## Slide 5. Related Work and Gap

### Slide title
Related Work and the Missing Layer

### On-slide text
- cryo-EM platforms such as cryoSPARC, RELION, and Warp automate important processing modules and preprocessing steps
- Deep learning methods such as Topaz, cryoDRGN, and e2gmm improve particle picking and heterogeneity analysis
- Recent agentic AI research suggests that scientific workflows can be assisted by planning-and-tool-using models
- Gap: these directions are still disconnected, and there is no unified agent layer for dataset-aware, state-aware, executable EM workflow decision making

### Speaker note
The structure of the related work should be presented in three layers. First, software systems such as cryoSPARC, RELION-3, and Warp improve automation inside the processing stack. Second, machine learning methods such as Topaz, cryoDRGN, and e2gmm improve strong local subtasks like particle picking and heterogeneity modeling. Third, recent agentic-science surveys argue that tool-using scientific assistants are becoming feasible. Our gap is between these layers: we want a domain-specific decision controller that reads dataset context and workflow state, then decides the next executable processing action.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryoagent_research_gap_map.svg`.
- Optional manual enhancement: add small logos or citations for cryoSPARC, Warp, Topaz, cryoDRGN, and MCP.

## Slide 6. Overall System Idea

### Slide title
System-Level Idea

### On-slide text
- Separate decision generation from execution control
- Model: predicts the next workflow action from V2 context
- MCP server: builds state, validates output, plans execution, runs jobs, monitors status
- JSON is the contract between the two layers

### Speaker note
This split is important. It lets us keep the model focused on reasoning and planning while giving the system a controlled execution layer that can enforce safety, approval rules, and tool constraints. The design should later extend to tools beyond cryoSPARC.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg`.

## Slide 7. MCP Server Framework

### Slide title
Outer MCP Server Architecture

### On-slide text
- Implemented repository: `/ssd1/linweifan/cryosparc_agent`
- Five implemented layers:
- CryoSPARC access
- workflow state extraction
- job metadata and execution
- decision alignment
- model input and MCP tools

### Speaker note
The MCP server is already beyond design discussion. It can connect to CryoSPARC, extract workspace state, construct model input, validate model output, and in guarded cases submit and monitor real jobs. Conceptually, this outer controller can later be expanded to cover other EM data-processing tools as well.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryosparc_mcp_architecture_preview.png`.
- Backup: use `ppt_assets_group_meeting_20260706/cryosparc_mcp_architecture.svg` if you want to edit labels.

## Slide 8. Closed-Loop Workflow

### Slide title
From State to Decision to Action

### On-slide text
1. MCP collects dataset metadata and workflow state
2. Model predicts `forward`, `branch`, `rollback`, or `stop`
3. MCP adapts and validates the decision
4. MCP creates a plan, executes when allowed, and monitors the result
5. The result becomes the next round's input

### Speaker note
This is the actual operational loop. One of the strengths of the current prototype is that this loop has already been smoke-tested with real cryoSPARC jobs, not only offline JSON examples.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryoagent_closed_loop_diagram.svg`.

## Slide 9. Data and Input Design

### Slide title
Unified V2 Input Schema

### On-slide text
- Data sources: EMDB XML, workflow JSON, runtime logs, dataset labels, live workspace state
- Top-level structure:
- `dataset_info`
- `current_state`
- Supports both known-workflow and no-workflow scenarios

### Speaker note
The V2 schema is one of the core design decisions. We do not want one model for replaying known workflows and another for unseen datasets. Instead, we keep the interface unified and allow `known_workflow_steps` to be present or null.

### Suggested image
- Use a small schema screenshot from the V2 documentation, or build a clean two-block slide.
- Optional source: [CRYOAGENT_V2_DATA_SCHEMA.md](/home/lisongyang/cryoagent/CRYOAGENT_V2_DATA_SCHEMA.md)

## Slide 10. Model Training Strategy

### Slide title
Model Adaptation Plan

### On-slide text
- Base model: `Qwen3.6-27B`
- SFT teaches workflow decision format and behavior
- CPT injects broader cryo-EM domain knowledge from literature
- Planned comparison:
- `Base -> SFT`
- `Base -> CPT -> SFT`

### Speaker note
SFT teaches the model how to act in the task, while CPT is intended to improve what the model knows about cryo-EM methods and workflows. The expected benefit is better generalization, especially in the no-workflow setting.

### Suggested image
- Use a two-stage pipeline diagram.
- Current recommendation: use a simple text-plus-arrow build directly in PPT, or use `cryoagent_closed_loop_diagram.svg` cropped to the model area.

## Slide 11. Current Progress

### Slide title
Current Progress: Result Snapshot

### On-slide text
- `No-workflow validation (137 samples)`: `99.27%` valid JSON, `96.35%` decision-type accuracy, `39.42%` exact action+parameter match
- `Known-workflow upper bound (152 samples)`: `98.68%` exact match in a strong-context setting
- `MCP live validation`: real cryoSPARC jobs `J36`, `J42`, and `J59` were created and monitored successfully
- `Human-in-the-loop handling`: interactive job `J37` was correctly paused with `ready_for_model = false`

### Speaker note
The most important message is that we now have both model-side and system-side evidence. On the model side, the current no-workflow result shows strong JSON stability and strong decision-type accuracy, although exact action and parameter matching still lag behind. On the system side, the MCP server has already validated real job creation, queueing, monitoring, and pause behavior for interactive steps.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryosparc_mcp_architecture_简版_preview.png` or a custom summary slide with one metric box for model results and one case-study box for MCP results.

## Slide 12. Differentiation and Innovation

### Slide title
What Is Distinct About This Project

### On-slide text
- Workflow-level decision control rather than isolated cryo-EM subtasks
- Unified schema for known-workflow reuse and no-workflow exploration
- Runtime-aware state construction from logs and workflow records
- MCP server as execution, monitoring, validation, and safety layer
- `CPT -> SFT` as a domain-aware adaptation path

### Speaker note
The novelty here is not a single algorithmic trick. It is the combination of a decision model, a structured state interface, and a controlled execution layer designed for real workflows. cryoSPARC is the current validation target, but the system framing is intentionally more general.

### Suggested image
- Use `ppt_assets_group_meeting_20260706/cryoagent_research_gap_map.svg`.

## Slide 13. Remaining Challenges

### Slide title
What Still Needs Work

### On-slide text
- Exact node-level and branch decisions remain harder than coarse action selection
- Parameter prediction still needs stronger constraints and validation
- Interactive cryoSPARC steps require a cleaner human-in-the-loop protocol
- The MCP execution policy should evolve from smoke-test rules to production governance

### Speaker note
This is a good place to be candid. The system direction looks viable, but reliability at the fine-grained execution level is still the main challenge. That is especially true when multiple candidate actions are plausible or when parameters matter more than action type.

### Suggested image
- No complex image required. A four-box risk/challenge layout is enough.

## Slide 14. Next Steps and Discussion

### Slide title
Next Steps

### On-slide text
- Run the first CPT experiment and compare against the current SFT baseline
- Expand MCP integration from smoke tests to stable closed-loop runs on representative datasets
- Evaluate both decision quality and execution success
- Refine the human-in-the-loop boundary for interactive steps

### Speaker note
The next phase is about turning the current prototype into a more reliable experimental system. That means better model adaptation, better execution evaluation, and clearer integration rules for human-interactive jobs.

### Suggested image
- A simple roadmap arrow or milestone strip.
- Current recommendation: use a text-led slide with minimal graphics.
