# Agent Hotfix: Missing Pick Overlay

## Fixed Failure

The 2026-09-15 handoff could raise `KeyError: 'contact_sheet'` after Blob Picker
when `get_pick_inspection_visual_context` returned an error payload instead of an
image overlay. The runner now treats visual overlays as optional evidence:

- A valid `contact_sheet.data_url` is attached to the multimodal model message.
- A missing or failed overlay is recorded as `visual_attachment_status=unavailable`.
- The model continues from the live CryoSPARC state, candidate actions, and any
  structured visual statistics; it does not terminate or request human input only
  because an overlay could not be generated.

The same guard is present in both `autonomous_mcp_closed_loop.py` and
`model_direct_runner.py`.

## Deployment

Use `cryoagent-agent-source-20260917-hotfix.tar.gz` instead of the 2026-09-15
archive. Extract it to a new directory or replace the handoff source tree, then
restart the runner. A process already running the old Python files must be stopped
and restarted for the fix to take effect.

When investigating a missing image, keep the affected round directory. Its
`visual_context.json`, `model_input.json`, `candidate_actions.json`, and
`model_messages.json` retain the structured evidence and the exact non-fatal
visual error.

## Verification

The handoff environment passed 81 unit tests, including regression cases for both
missing and valid `contact_sheet` payloads.
