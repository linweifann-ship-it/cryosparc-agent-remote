# MCP V24 Multi-Model Benchmark Plan

## Boundaries

- Schema V24 is the only model-facing contract.
- Model input is `schema_version="2.1"` workflow decision context.
- Model output is `schema_version="3.0"` minimal_v3 JSON.
- Providers live in the runner, not in the MCP server.
- The model only chooses `decision_type` and `selected_actions[].job_type/parameters`.
- MCP builds objective context, validates output, resolves deterministic internal connections, and returns objective results.
- Runner calls providers, calls MCP, isolates runs, writes append-only logs, handles fixed infrastructure retries, and stops.
- Codex only starts, observes, and summarizes benchmark runs.

## Forbidden Model-Facing Behavior

- Do not expose explicit `candidate_actions` to the model.
- Do not expose or restore `failure_context` or `retry_guidance`.
- Do not accept model-supplied `connections`, `workflow_node_id`, `reason`, `confidence`, `evidence`, `rollback_target`, or `branch_plan`.
- Do not repair, migrate, trim, extract, or otherwise rewrite model output before V24 validation.
- Do not correct model decisions from historical successful workflows.
- Do not create real CryoSPARC jobs in validation-only benchmark runs.

## Provider Layer

The benchmark runner supports:

- `local`: existing local Qwen worker.
- `openai`: OpenAI Responses API using `OPENAI_API_KEY`, `store=false`, and V24 JSON schema output format.
- `dashscope_qwen`: DashScope Qwen API using `DASHSCOPE_API_KEY`; raw Qwen can use OpenAI-compatible chat completions, and fine-tuned Qwen must record the exact model or checkpoint ID.

All providers return only raw visible output, sanitized response metadata, exact model ID when available, usage, latency, request ID, and error state.

## Run Isolation And Connections

- Every run has a `run_id`.
- `allowed_job_uids` is the start/current job plus jobs created by the same run.
- Every round asserts that model input and prompt messages contain only allowed run context.
- `known_workflow_steps` must be `null`.
- Internal candidate actions may be saved for evidence only as `internal_candidate_actions.json` with `not_sent_to_model=true`.
- Connection resolution is deterministic:
  - exactly one valid source: connect internally and record it;
  - zero sources: `connection_resolution_failed`;
  - multiple sources: `ambiguous_connection`.

## Benchmark Logging

Each run writes:

- `benchmark_manifest.json`
- `run_config.json`
- `launcher.log`
- `model_client.log`
- `events.jsonl`
- `metrics.json`
- `summary.json`
- `round_001/` evidence files

The benchmark manifest records benchmark ID, run ID, provider, model label, exact model ID, fine-tune checkpoint, schema version, prompt hash, dataset hash, start node, runner/MCP commit, config hash, temperature, seed, max rounds, and dirty state. Unknown values are `null`.

The compare tool is read-only and outputs:

- `benchmark_summary.json`
- `benchmark_table.csv`
- `benchmark_report.md`

Runs with different start node, schema, prompt hash, dataset hash, commits, max rounds, or approval policy are marked `not_directly_comparable`.

## Test Policy

Implementation must pass mock/validation-only tests first:

- provider success and API failure cases;
- strict JSON parsing and V24 forbidden fields;
- run isolation for shared workspace scenarios;
- unique, zero, and ambiguous connection outcomes;
- validation-only runner creates no CryoSPARC job;
- benchmark comparison is repeatable and does not mutate raw logs;
- API keys are not written to benchmark logs.
