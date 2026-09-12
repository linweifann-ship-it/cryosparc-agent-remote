# KB-Guided Model Decision Flow

The current API closed-loop runner now supports this sequence per decision round:

1. The runner obtains the live CryoSPARC state and live candidate actions.
2. The model receives that state and a read-only kb_* tool catalog.
3. The model is required by default to request at least one KB tool using standard OpenAI-compatible tool_calls; subsequent KB calls remain optional.
4. The runner executes only the requested read-only KB MCP tool and appends the result as a role=tool message.
5. The model produces the final V2 JSON decision.
6. The existing V2 validation and CryoSPARC execution path remains authoritative.

## Exposed tools

The current server exposes:

- kb_get_decision_context: compact bundle combining dataset summary, similar cases, observed next steps, candidate job documentation, and failure evidence.
- kb_search_cryoem_kb: full-text search over workflow, documentation, logs, metrics, and annotations.
- kb_get_dataset_summary, kb_get_workflow, kb_get_maps: exact historical dataset evidence.
- kb_find_similar_cases: non-excluded historical cases for generalization.
- kb_get_next_steps: empirical transitions after a job type.
- kb_get_job_doc: official CryoSPARC job documentation.
- kb_get_failures, kb_get_images, kb_get_manual_annotations: failure, image, and expert evidence.

All are read-only and prefixed with kb_. No job creation or execution tool is included in the model tool catalog.

## Leakage control

EMPIAR-10025 is normalized to KB dataset ID 10025. An exact workflow is only reported when the KB actually contains that dataset. Similar-case retrieval excludes records marked exclude_from_case_retrieval=1. Historical Job IDs are evidence only; the model must choose from the current live candidate list.

Round logs now include kb_tool_calls.json and the full model request/response trace. The maximum is controlled by --max-kb-tool-calls (default: 4).

## Configuration

The bridge defaults to:

/hdd1/huangjianhua/agent/data0/kb_build_v2

Override it without changing code:

export CRYOAGENT_KB_ROOT=/path/to/kb_build_v2

The API backend enables tool calling automatically and defaults to --kb-tool-policy required. Use --kb-tool-policy auto only for compatibility experiments with providers that do not support required tool choice. The local Qwen worker keeps the previous plain-JSON path for now; it does not yet autonomously issue MCP tool calls.
