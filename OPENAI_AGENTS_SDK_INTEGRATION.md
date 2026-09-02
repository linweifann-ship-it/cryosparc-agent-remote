# OpenAI Agents SDK Integration Notes

## Existing implementation checked

- MCP server entry: `cryosparc_agent_remote/cryosparc_mcp_server.py`; remote deployment keeps the same entry at `/ssd1/linweifan/cryosparc_agent/cryosparc_mcp_server.py`.
- MCP tool definitions: `cryosparc_mcp_server.py` exposes health, workflow state, candidate action, validation, execution, model input, and result package tools through `FastMCP`.
- Current model client / runner: `cryosparc_agent_remote/model_direct_runner.py` implements local model and OpenAI-compatible Chat Completions calls; `cryosparc_agent_remote/autonomous_mcp_closed_loop.py` runs the previous custom closed loop.
- Current closed-loop script: `scripts/autonomous_mcp_closed_loop.py` delegates state, validation, execution, and job result waiting to the MCP server via stdio.
- Schema/state/result package: `schemas.py`, `model_input_builder.py`, `workflow_state.py`, `job_result.py`, and `MCP_SERVER_IO_SCHEMA_V24.md`.
- Prompt cache implementation: `model_direct_runner.py` accepts `prompt_cache_key` and `prompt_cache_options`; `autonomous_mcp_closed_loop.py` keeps static instructions before dynamic state and can mark explicit prompt cache breakpoints.
- CryoSPARC job flow: `action_registry.py` validates/adapts model decisions, `job_executor.py` calls `workspace.create_job(...)` and queues jobs, and `job_result.py` waits for terminal job states before returning model-facing observations.

## Added architecture

`GPT-5.6-sol -> OpenAI Agents SDK Agent/Runner -> existing MCP Server -> cryoSPARC -> Observation -> Agent`

The Agents SDK layer is implemented in `cryosparc_agent_remote/openai_agents_runner.py`.

No existing MCP tool implementation was copied into the SDK layer. The SDK layer starts the existing MCP server over stdio through `MCPServerStdio`, discovers the current tool list, and lets the Agent call those tools.

## MCP changes

No MCP server tool implementation was changed for this integration. The only MCP-related behavior added is client-side SDK connection code that uses the existing stdio server.

## Entry points

Single closed loop:

```bash
python scripts/run_openai_agents_closed_loop.py \
  --project P2 \
  --workspace W4 \
  --start-node J34 \
  --model gpt-5.6-sol \
  --run-id example
```

Independent benchmark:

```bash
python scripts/run_independent_benchmark.py \
  --runs 3 \
  --project P2 \
  --workspace W4 \
  --start-node J34 \
  --model gpt-5.6-sol
```

Set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.ofox.ai/v1
export OPENAI_MODEL=gpt-5.6-sol
```
