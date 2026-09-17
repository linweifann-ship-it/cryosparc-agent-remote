# ProSHADE Symmetry Evidence Integration

## Purpose

The autonomous CryoSPARC runner can optionally run [ProSHADE](https://github.com/michaltykac/proshade)
on a completed ab-initio or refinement volume.  It sends the resulting symmetry suggestion, axis,
and FSC score to the decision model as `symmetry_evidence`.

This is advisory evidence.  The agent must not overwrite an existing C1 refinement merely because
ProSHADE reports a non-C1 symmetry.  Instead, it should retain the C1 path and, when justified,
compare a symmetry-constrained refinement branch.

## Enabled Job Types

The MCP tool supports completed `homo_abinit`, `homo_refine_new`,
`nonuniform_refine_new`, and `refine_3D_new` jobs.  It locates the primary non-half volume,
downloads it to the local cache, and invokes ProSHADE.

## Installed Runtime

The executable is installed at:

```bash
/home/lisongyang/.local/proshade/install/bin/proshade
```

The integration uses this path by default. To override it for one shell or deployment:

```bash
export CRYOAGENT_PROSHADE_BINARY=/path/to/proshade
```

The local build source is under `/home/lisongyang/.local/proshade/src/proshade`.  To build with
the current server toolchain, ProSHADE was compiled with C++17 and Gemmi `v0.5.7`; these are build
compatibility details, not runtime requirements for the agent.

## Runner Usage

Add the following flags to an existing autonomous closed-loop command:

```bash
--proshade-symmetry \
--proshade-resolution-A 8.0 \
--proshade-timeout-seconds 1800
```

For maps whose centre is uncertain, add `--proshade-find-symmetry-center`. This is substantially
slower and should normally be reserved for a later confirmation run.

For example:

```bash
PYTHONPATH=/home/lisongyang/cryoagent:/home/lisongyang/cryoagent/cryosparc_agent \
/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python -u \
  /home/lisongyang/cryoagent/cryosparc_agent/autonomous_mcp_closed_loop.py \
  ...existing runner arguments... \
  --proshade-symmetry --proshade-resolution-A 8.0
```

## Output And Audit Trail

Each eligible round writes `symmetry_evidence.json` in its round directory. A successful result
contains `recommended_symmetry` (for example `C3` or `D7`), `primary_axis`, `axis_height`,
`average_fsc`, and a coarse `confidence` label.  Failures are also structured and non-fatal, with
an `error_code`; the normal model decision continues without a symmetry recommendation.

Downloaded maps are cached under `/home/lisongyang/cryoagent/logs/proshade` by default. Override
this with `CRYOAGENT_PROSHADE_CACHE_DIR` when a different scratch or storage location is needed.

## Licensing Note

Please review ProSHADE and its dependency licenses before redistributing a packaged agent binary.
The upstream project documents that the default dependency combination can impose GPL obligations
on distributed binaries.
