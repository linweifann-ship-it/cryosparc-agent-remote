# CryoAgent MCP Handoff Guide (2026-09-15)

## 1. Scope and current status

This package contains the current CryoSPARC workflow-agent source code.  The
main runtime is `cryosparc_agent/autonomous_mcp_closed_loop.py`, which runs an
autonomous loop:

1. Read the live CryoSPARC project/workspace state.
2. Generate executable candidate jobs from the CryoSPARC registry and DAG.
3. Optionally retrieve read-only knowledge-base evidence.
4. Call either an OpenAI-compatible API model or a local Qwen model.
5. Validate the JSON decision, create and queue the selected CryoSPARC job.
6. Wait for the job result and repeat until `--max-rounds` or a stop condition.

The code has been exercised on the local CryoSPARC deployment and has unit
tests for its decision, candidate, retry, and MCP logic.  It is **not yet a
cross-version production release**: live CryoSPARC job parameters can differ
between CryoSPARC installations.  In particular, validate `inspect_picks_v2`
on the target deployment before launching a long unattended workflow.

The package intentionally excludes model weights/adapters, data, CryoSPARC
installation files, the external KB SQLite database, API keys, logs, and
CryoSift weights.  CryoSift is optional and disabled by default.

## 2. Package layout

```text
cryoagent/
  cryosparc_agent/                 # MCP server, runner, validation, tests
  test_inputs/                     # Example dataset JSON payloads
  MCP_SERVER_IO_SCHEMA_V23.md       # MCP input/output contract
  CRYOAGENT_V2_DATA_SCHEMA.md       # Dataset/current-state V2 schema
  KB_MODEL_TOOL_CALLING.md          # KB design notes
  CRYOAGENT_AGENT_HANDOFF_20260915.md
  requirements-agent-handoff.txt
```

Important entry points:

| Path | Purpose |
| --- | --- |
| `cryosparc_agent/autonomous_mcp_closed_loop.py` | Full continuous API/local-model closed loop |
| `cryosparc_agent/cryosparc_mcp_server.py` | Stdio MCP server exposing live CryoSPARC and KB tools |
| `cryosparc_agent/action_registry.py` | Candidate validation and execution-plan construction |
| `cryosparc_agent/job_executor.py` | Job creation, connections, queueing, resource routing |
| `cryosparc_agent/dynamic_candidates.py` | Runtime CryoSPARC registry candidates and input matching |
| `cryosparc_agent/vision_inputs.py` | Select 2D and Inspect Picks visual evidence extraction |
| `cryosparc_agent/tests/` | Offline unit tests |

## 3. Required environment

Use a Python environment that has at least the packages in
`requirements-agent-handoff.txt`.  The environment used for this snapshot was:

```text
Python 3.11
mcp 1.27.2
cryosparc-tools 5.0.3
pydantic 2.13.4
```

The runner starts the MCP server as a child process, so the `mcp` and
`cryosparc-tools` packages must be installed in `--server-python`'s environment.
For a local Qwen backend, `transformers`, `torch`, `peft`, and the model adapter
must be installed/available in `--model-python`'s environment.

### CryoSPARC access

The source defaults to `localhost:61000`.  Configure either a CryoSPARC
license or a user login in the shell that starts the runner:

```bash
export CRYOSPARC_EMAIL='agent@example.org'
export CRYOSPARC_PASSWORD='replace-with-password'
# Alternative: export CRYOSPARC_LICENSE_ID='...'
```

When CryoSPARC is on a remote host, establish an SSH tunnel first, for example:

```bash
ssh -N -L 61000:127.0.0.1:61000 <user>@<cryosparc-host>
```

The client automatically appends `localhost`, `127.0.0.1`, `admin`, and
`172.16.1.2` to `NO_PROXY`/`no_proxy`; it does not disable your general API
proxy.  For a different CryoSPARC host/port, update the defaults in
`cryosparc_agent/cryosparc_client.py` or run through a matching local tunnel.

### API model access

Do not put API secrets in this repository.  Use a protected environment file:

```bash
mkdir -p ~/.config/cryoagent
chmod 700 ~/.config/cryoagent
cat > ~/.config/cryoagent/api.env <<'EOF'
OPENAI_API_KEY=replace-with-secret
OPENAI_BASE_URL=https://api.example.org/v1
OPENAI_MODEL=provider/model-name
EOF
chmod 600 ~/.config/cryoagent/api.env
export CRYOAGENT_API_ENV_FILE=~/.config/cryoagent/api.env
```

Or export `OPENAI_API_KEY` directly.  Supply `--api-base` and `--api-model` on
the command line; the runner never writes the key to its report directory.

### Optional knowledge base

By default, the code expects the separate KB checkout at
`/hdd1/huangjianhua/agent/data0/kb_build_v2`.  For another machine, point to
the external KB root without copying its database into this package:

```bash
export CRYOAGENT_KB_ROOT=/path/to/kb_build_v2
```

If no KB is available, use `--kb-tool-policy disabled`.  Do not use the default
`required` mode without a working KB, because the model is instructed to make a
read-only KB call before deciding.

## 4. Preflight checks

Set these variables after unpacking the archive:

```bash
export PACKAGE_ROOT=/path/to/cryoagent
export AGENT_DIR="$PACKAGE_ROOT/cryosparc_agent"
export SERVER_PYTHON=/path/to/conda/env/bin/python
export PYTHONPATH="$PACKAGE_ROOT:$AGENT_DIR"
cd "$AGENT_DIR"
```

Run the offline test suite first:

```bash
"$SERVER_PYTHON" -m unittest discover -s tests -p 'test_*.py'
```

Then verify the target CryoSPARC API login and worker visibility:

```bash
"$SERVER_PYTHON" -c 'from cryosparc_client import cryosparc_client; print(cryosparc_client())'
"$SERVER_PYTHON" -c 'from cryosparc_cli_tools import cryosparc_status, cryosparc_version; print(cryosparc_status()); print(cryosparc_version())'
```

Before a real workflow, create a disposable project/workspace and test one
`inspect_picks_v2` candidate against the live registry.  Preserve the returned
`validation.json` and `execution.json` if it fails; these reveal whether the
failure is local decision validation, unsupported CryoSPARC parameters, input
connections, or queue configuration.

## 5. Dataset JSON

Provide a JSON file with facts known before processing.  Avoid putting future
job results in this file.  Example `dataset.json`:

```json
{
  "empiar_id": "EMPIAR-10025",
  "input_type": "micrographs",
  "macromolecules_type": "protein complex",
  "available_input_files": {
    "micrograph_blob_paths": "/data/EMPIAR-10025/*.mrc"
  },
  "psize_A": 0.6575,
  "accel_kv": 300,
  "cs_mm": 2.7,
  "total_dose_e_per_A2": 53.0,
  "target_resolution_A": 2.8
}
```

For movie datasets, use `movie_blob_paths` and `input_type: "movies"`.  The
agent passes acquisition facts as import defaults and relies on live job status
for all downstream decisions.  More field details are in
`MCP_SERVER_IO_SCHEMA_V23.md` and `CRYOAGENT_V2_DATA_SCHEMA.md`.

## 6. Start a continuous API-backed workflow

The following uses one process for the full loop.  It creates and queues real
CryoSPARC jobs, so begin with a disposable project/workspace and a low
`--max-rounds` value.

```bash
cd "$AGENT_DIR"
mkdir -p "$PACKAGE_ROOT/runs"

nohup "$SERVER_PYTHON" -u autonomous_mcp_closed_loop.py \
  --project P1 \
  --workspace W1 \
  --dataset-json-file /absolute/path/dataset.json \
  --backend api \
  --api-base "$OPENAI_BASE_URL" \
  --api-model "$OPENAI_MODEL" \
  --api-key-env OPENAI_API_KEY \
  --kb-tool-policy disabled \
  --max-rounds 4 \
  --max-new-tokens 4096 \
  --wait-timeout-seconds 43200 \
  --poll-interval-seconds 60 \
  --output-dir "$PACKAGE_ROOT/runs/api_smoke" \
  > "$PACKAGE_ROOT/runs/api_smoke.log" 2>&1 &
echo $!
```

For KB-enabled execution, set `CRYOAGENT_KB_ROOT` and change the policy to
`--kb-tool-policy required` (or `auto`).  For visual Class 2D evidence add
`--vision`.  CryoSift is intentionally opt-in: only add `--cryosift` after its
separate runner and weights have been configured.

The runner writes `summary.json`, one `round_XX/` directory per decision, raw
model messages, candidate actions, validation/execution results, and a
checkpoint JSON.  Do not delete these while the process is active.

## 7. Resume after interruption

Find the latest checkpoint under the prior run directory, then start a new
process with the same project/workspace and dataset inputs:

```bash
"$SERVER_PYTHON" -u autonomous_mcp_closed_loop.py \
  --project P1 --workspace W1 \
  --dataset-json-file /absolute/path/dataset.json \
  --backend api --api-base "$OPENAI_BASE_URL" --api-model "$OPENAI_MODEL" \
  --api-key-env OPENAI_API_KEY \
  --kb-tool-policy disabled \
  --resume-checkpoint /absolute/path/to/checkpoint.json \
  --output-dir "$PACKAGE_ROOT/runs/api_resume"
```

Do not resume by merely changing `--current-node` unless intentionally starting
a new branch.  The checkpoint retains the preceding job/result state and avoids
repeating a completed action.

## 8. Inspect Picks compatibility checklist

`inspect_picks_v2` is special because CryoSPARC may advertise it as interactive
even though this agent uses an automated threshold mode.  The code overrides
the interactive flag for this job and requires an explicit automated filtering
choice.  Common failure categories are:

| Symptom | Likely cause | Immediate debugging action |
| --- | --- | --- |
| `unknown parameter` / server parameter validation error | Local parameter name does not exist in the target CryoSPARC version | Save `candidate_actions.json`; compare its live registry `parameter_template` with the rejected parameter list. |
| `inspect_parameters_required` | Model emitted no threshold or only `do_auto_cluster=false` | Inspect `model_decision.json`; re-run with an explicit NCC or lower Power threshold. |
| Job created but not queued | Older worker lane setup or external scheduler error | Inspect `execution.json`, job status in UI, and scheduler logs; Inspect Picks itself is CPU/non-GPU. |
| Interactive UI prompt | Target version still requires manual curation despite automatic metadata override | Treat that CryoSPARC version as unsupported for unattended Inspect Picks until a version-specific adapter is added. |

The relevant source locations are `job_specs.py`, `dynamic_candidates.py`,
`cryosparc_mcp_server.py`, and `job_executor.py`.  Please share the exact
CryoSPARC error text plus the corresponding `candidate_actions.json`,
`model_decision.json`, `validation.json`, and `execution.json` when reporting a
failure.

## 9. Operational cautions

- This runner can create real jobs.  Use a separate test project and workspace.
- The API endpoint, CryoSPARC master, worker queues, and KB must remain reachable
  for an unattended run.  Retry logic handles transient failures but not an
  incompatible job schema or a persistent service outage.
- Keep API keys out of shell history, repository files, reports, and archives.
- GPU lane choice is automatic by default.  To force a configured lane, set
  `CRYOAGENT_GPU_LANE`; to use scheduler capacity selection, leave it unset.
- Quality rollback is opt-in: `export CRYOAGENT_ENABLE_QUALITY_ROLLBACK=1`.
  Validate this policy on a disposable project before production use.
