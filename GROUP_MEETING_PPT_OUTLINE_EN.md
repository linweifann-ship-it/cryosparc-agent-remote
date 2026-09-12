# CryoAgent Group Meeting PPT Outline

## Slide 1. Title
- CryoAgent: An Automated Electron Microscopy Data Processing Assistant
- First project progress report
- Presenter, group, date

## Slide 2. Background and Motivation
- Modern cryo-EM can acquire large datasets quickly, but downstream data processing remains a major bottleneck
- The pipeline is multi-step and iterative, with noisy data, unknown poses, structural heterogeneity, and expensive computation
- Important subtasks and workflow choices still often require substantial expert effort

## Slide 3. Problem Statement
- Goal: build an agent system that reads dataset context and current workflow state, then decides the next data-processing action and parameters
- Target output: structured JSON that can be validated and executed by an external controller
- Long-term objective: closed-loop workflow automation rather than isolated prediction

## Slide 4. Why This Problem Is Challenging
- Workflow planning is sequential, state-dependent, and error-sensitive
- Decisions require both scientific context and runtime execution evidence
- The system must support known workflow reuse, unseen-dataset exploration, and human-interactive steps

## Slide 5. Related Work and Gap
- cryo-EM platforms such as cryoSPARC, RELION, and Warp automate important processing modules and preprocessing steps
- Deep learning methods such as Topaz, cryoDRGN, and e2gmm improve particle picking and heterogeneity analysis
- Recent agentic AI research suggests that scientific workflows can be assisted by planning-and-tool-using models
- Gap: these directions are still disconnected, and there is no unified agent layer for dataset-aware, state-aware, executable EM workflow decision making

## Slide 6. Overall System Idea
- Treat electron microscopy data processing as a model-guided decision loop wrapped by a controlled MCP execution layer
- MCP server handles tool access, workflow-state extraction, action validation, execution planning, job submission, monitoring, and result packaging
- The model focuses on next-step decision generation from the V2 context payload
- Structured JSON is the stable contract between the two layers

## Slide 7. MCP Server Framework
- Current repository: `/ssd1/linweifan/cryosparc_agent`
- Implemented layers: CryoSPARC access, workflow state, job metadata/execution, decision alignment, and model input/MCP tools
- Key modules include `workflow_state.py`, `model_input_builder.py`, `v2_decision_adapter.py`, `action_registry.py`, `job_executor.py`, and `job_result.py`
- Safety policy: dry-run by default, approval gates for interactive jobs, rollback, unknown jobs, and high-GPU actions

## Slide 8. Closed-Loop Workflow
- MCP builds a V2 model input from dataset metadata, workflow history, and current job result
- Model returns a compact decision such as `forward`, `branch`, `rollback`, or `stop`
- MCP adapts the V2 decision to the internal execution schema, validates it against candidate actions, and creates a plan
- After execution or human interaction, MCP packages the completed result and starts the next decision round

## Slide 9. Data and Input Design
- Data sources include EMDB XML, workflow JSON, runtime logs, dataset labels, and live CryoSPARC workspace state
- V2 schema is centered on `dataset_info` and `current_state`
- `known_workflow_steps` can be filled by retrieval when a dataset match exists, or left null for exploratory planning
- `last_node_info` summarizes outputs, metrics, warnings, runtime, connections, and recent nodes

## Slide 10. Model Training Strategy
- Base model: `Qwen3.6-27B`
- SFT teaches structured workflow decision behavior
- CPT injects broader cryo-EM domain knowledge from literature
- Main comparison planned: `Base -> SFT` vs `Base -> CPT -> SFT`

## Slide 11. Current Progress
- No-workflow validation on `137` samples reached `99.27%` valid JSON and `96.35%` decision-type accuracy, with `39.42%` exact action+parameter match
- In a workflow-aware setting, an earlier run reached `98.68%` exact match on `152` samples, showing a strong upper-bound result
- MCP side: server tools, V2 payload builder, decision adapter, action validation, guarded execution, and job result packaging are implemented
- Closed-loop tests have already created and monitored real cryoSPARC jobs such as `J36`, `J42`, and `J59`, and correctly paused at interactive step `J37`

## Slide 12. Differentiation and Innovation
- Workflow-level decision control rather than single-module cryo-EM prediction
- Unified schema for known-workflow reuse and no-workflow exploration
- Structured log/workflow-to-state conversion for runtime-aware decisions
- MCP server as the execution, validation, monitoring, and safety layer around the model
- `CPT -> SFT` adaptation for domain-aware workflow planning

## Slide 13. Remaining Challenges
- Exact node-level and branch decisions are harder than high-level action selection
- Parameter prediction needs stronger validation and possibly tighter tool-side defaults
- Interactive cryoSPARC jobs require a clear human-in-the-loop protocol
- MCP execution policy needs to mature from smoke-test safety rules into production governance

## Slide 14. Next Steps and Discussion
- Run the first CPT experiment and compare with the current SFT baseline
- Expand MCP integration from smoke tests to a stable closed-loop workflow on representative datasets
- Add evaluation metrics that cover both model decision quality and MCP execution success
- Discussion: is the model/MCP boundary and human-in-the-loop strategy reasonable?
