# Related Work and Innovation Notes

## Literature Snapshot

### 1. Automated cryo-EM processing software
- `cryoSPARC` accelerated unsupervised cryo-EM structure determination by introducing fast optimization algorithms and a user-friendly program. It explicitly identified expert intervention and long processing cycles as a major bottleneck in cryo-EM data processing.
- `Warp` automated real-time preprocessing, including motion correction, defocus estimation, particle picking, denoising, and real-time quality monitoring.
- `RELION-3` and related workflow/scheme mechanisms improved high-resolution cryo-EM processing and made parts of the pipeline more scriptable and reproducible.
- These systems automate important computational steps, but they still usually assume that humans or predefined workflows decide which processing step to run next and how to respond to dataset-specific failures.

### 2. Deep learning for cryo-EM subtasks
- `Topaz` uses positive-unlabeled learning for particle picking with limited labels and no labeled negatives.
- `cryoDRGN` reconstructs continuous heterogeneous 3D density distributions from single-particle cryo-EM data.
- `e2gmm` models conformational variability using a deep-learning-based Gaussian mixture representation.
- Recent work continues to improve particle picking, heterogeneity analysis, reconstruction, and map interpretation.
- The gap is that these methods target specialized analysis modules rather than the full decision layer that coordinates an end-to-end cryoSPARC workflow.

### 3. LLM agents for scientific workflows
- Recent surveys on autonomous scientific discovery describe LLM agents as systems that combine literature knowledge, planning, tool use, experiment execution, and iterative refinement.
- Scientific agents are moving from task assistance toward closed-loop execution, but most examples are in chemistry, materials, coding, molecular simulation, or general research automation.
- HPC-oriented LLM-agent work shows that agents can be connected to compute resources and workflow execution engines.
- Workflow-provenance agents show the value of querying and interpreting runtime records, but usually focus on analysis of past provenance rather than deciding the next cryo-EM processing action.

### 4. MCP and tool-interface ecosystem
- MCP provides an open protocol for connecting AI applications with external tools, data sources, and workflows.
- Its value for this project is not just model calling; it gives us a natural place to implement retrieval, state construction, validation, execution, logging, and safety checks.
- Recent MCP security and measurement studies also suggest that production deployment should treat the MCP server as a controlled execution layer, not a thin pass-through wrapper.


## Current MCP Server Implementation Snapshot

Repository inspected: `/ssd1/linweifan/cryosparc_agent`

The MCP server is already implemented as a five-layer prototype rather than only an interface proposal:

- CryoSPARC access layer: `cryosparc_client.py` and `cryosparc_cli_tools.py` connect to CryoSPARC services and CLI tools.
- Workflow state layer: `workflow_state.py` extracts a normalized workspace DAG with nodes, edges, inputs, outputs, running nodes, and failed nodes.
- Job metadata and execution layer: `job_specs.py`, `job_executor.py`, and `job_result.py` define supported job types, generate execution plans, submit guarded jobs, monitor status, and package completed results.
- Decision alignment layer: `schemas.py`, `action_registry.py`, and `v2_decision_adapter.py` validate model decisions and adapt compact V2 decisions into the internal execution contract.
- Model input and MCP tool layer: `model_input_builder.py`, `known_workflow_retriever.py`, and `cryosparc_mcp_server.py` build V2 context payloads and expose MCP tools.

Important implementation status:

- The server can build V2 model-facing payloads from live CryoSPARC workspace state.
- The server supports `forward`, `branch`, `rollback`, and `stop` decision types through validation/adaptation.
- Execution is conservative by default: dry-run first, with approval gates for interactive jobs, rollback, unknown jobs, and high GPU counts.
- Closed-loop reports show successful real job creation/monitoring, including a model decision after `select_2D`, live creation of `class_2D_new`, GPU-lane submission, completion monitoring, and recognition of an interactive `select_2D` step requiring human action.

This changes the project framing: the MCP server is not future infrastructure only. It is already a working orchestration prototype, while the model is the decision module inside that controlled execution loop.

## PPT-Ready: Related Work and Gap

- Existing cryo-EM platforms such as `cryoSPARC`, `RELION`, and `Warp` have significantly improved automation for reconstruction, preprocessing, and quality monitoring.
- Deep learning methods such as `Topaz`, `cryoDRGN`, and `e2gmm` further automate important cryo-EM subtasks, including particle picking and structural heterogeneity analysis.
- Recent LLM-agent research shows the potential of tool-using scientific agents, especially for planning, execution, and iterative refinement.
- However, these directions remain separated: cryo-EM tools automate individual processing modules, while general scientific agents rarely model cryoSPARC-specific workflow state and actions.
- The missing layer is a dataset-aware and state-aware decision controller that can choose the next cryoSPARC action, set parameters, and interact with an execution system in a closed loop.

## PPT-Ready: Differentiation and Innovation

- We focus on workflow-level decision making, not only isolated cryo-EM subtasks such as particle picking, reconstruction, or heterogeneity modeling.
- We design a unified decision schema that supports both known-workflow reuse and no-workflow exploratory planning.
- We convert runtime logs and workflow records into structured model-readable state, allowing decisions to depend on the current processing result rather than only static metadata.
- We separate responsibilities between an MCP execution layer and a domain-adapted decision model: the MCP server handles retrieval, validation, execution, and logging, while the model predicts the next action and parameters.
- We combine CPT and SFT so that the model can learn both cryo-EM domain knowledge and executable workflow decision behavior.

## Suggested Slide Replacement

### Slide 5. Related Work and Gap
- Cryo-EM software has automated many processing modules, but workflow planning still relies heavily on expert choices.
- Deep learning methods improve particle picking, reconstruction, and heterogeneity analysis, but usually do not control the full workflow.
- LLM agents can plan and use tools, but existing scientific agents are rarely specialized for cryoSPARC workflow state and execution.
- Gap: there is no unified agent layer for dataset-aware, state-aware, executable cryoSPARC decision making.

### Slide 11. Differentiation and Innovation
- Workflow-level decision control rather than single-module cryo-EM prediction.
- Unified schema for known-workflow reuse and no-workflow exploration.
- Structured log-to-state conversion for runtime-aware decisions.
- MCP server as the execution, validation, and safety layer around the model.
- `CPT -> SFT` adaptation for domain-aware workflow planning.

## Key References

- Punjani et al., "cryoSPARC: algorithms for rapid unsupervised cryo-EM structure determination", Nature Methods, 2017. https://www.nature.com/articles/nmeth.4169
- Tegunov and Cramer, "Real-time cryo-electron microscopy data preprocessing with Warp", Nature Methods, 2019. https://www.nature.com/articles/s41592-019-0580-y
- Bepler et al., "Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs", Nature Methods, 2019. https://www.nature.com/articles/s41592-019-0575-8
- Zhong et al., "CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks", Nature Methods, 2021. https://www.nature.com/articles/s41592-020-01049-4
- Chen and Ludtke, "Deep learning-based mixed-dimensional Gaussian mixture model for characterizing variability in cryo-EM", Nature Methods, 2021. https://www.nature.com/articles/s41592-021-01220-5
- Wei et al., "From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery", arXiv, 2025. https://arxiv.org/abs/2508.14111
- Zhou et al., "Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics", arXiv, 2025. https://arxiv.org/abs/2510.09901
- Ma et al., "Connecting Large Language Model Agent to High Performance Computing Resource", arXiv, 2025. https://arxiv.org/abs/2502.12280
- Model Context Protocol documentation. https://modelcontextprotocol.io/docs/getting-started/intro
- Anthropic, "Introducing the Model Context Protocol", 2024. https://www.anthropic.com/news/model-context-protocol
