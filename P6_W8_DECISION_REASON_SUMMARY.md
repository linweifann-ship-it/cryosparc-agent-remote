# P6/W8 EMPIAR-11280 Raw Model Decision Reasons

This document reproduces the model `reason` fields from the P6/W8 EMPIAR-11280 logs. The reason text is copied directly from `model_decision.json`; it is not summarized or rewritten. Failed submissions and `request_input` turns are included.

Dataset: `EMPIAR-11280`; project `P6`; workspace `W8`; 3,657 TIFF movie files.

## Initial import attempt

### run_20260907T151436Z / round_01

`action=import_movies`, `decision_type=forward`, `confidence=0.96`

```text
Import the available TIFF movies to initialize the CryoSPARC workflow. Acquisition metadata not supplied by the live state is left at job defaults rather than invented.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_debug/run_20260907T151436Z/round_01/model_decision.json`

## Initial preprocessing

### run_20260907T152918Z / round_01

`action=patch_motion_correction_multi`, `decision_type=forward`, `confidence=0.98`

```text
Run Patch Motion Correction on the successfully imported dose-fractionated movies before downstream CTF estimation. This follows both the authoritative live workflow recommendation and the dominant historical transition while preserving all acquisition metadata from J94.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_retry_nocache_j94/run_20260907T152918Z/round_01/model_decision.json`

## CTF recovery

### run_20260907T214756Z / round_01

`action=patch_ctf_estimation_multi`, `decision_type=forward`, `confidence=0.84`

```text
Proceed with the preferred standard CTF-estimation stage. The isolated incomplete exposure does not justify rerunning motion correction for the full dataset, and the live candidate provides a validated required input connection from J94.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_candidate_fix/run_20260907T214756Z/round_01/model_decision.json`

## Pre-picking recovery

### run_20260907T215019Z / round_01

`action=patch_ctf_estimation_multi`, `decision_type=forward`, `confidence=0.94`

```text
Proceed with Patch CTF Estimation on the completed micrograph output. The isolated incomplete exposure does not justify rerunning motion correction for the full dataset, and the available motion-corrected micrographs are sufficient for the standard next stage.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_type_fix/run_20260907T215019Z/round_01/model_decision.json`

### run_20260907T215019Z / round_02

`action=blob_picker_gpu`, `decision_type=forward`, `confidence=0.89`

```text
Proceed with reference-free initial picking from the completed CTF-estimated micrographs. A broad 180–240 Å blob range is appropriate for the large heterotetrameric NMDA receptor and can generate initial particles for downstream inspection and 2D classification without requiring unavailable templates.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_type_fix/run_20260907T215019Z/round_02/model_decision.json`

## Inspect Picks recovery

### run_20260908T030244Z / round_01

`action=inspect_picks_v2`, `decision_type=forward`, `confidence=0.96`

```text
Run the mandatory pick-inspection gate with explicit conservative NCC and Power lower thresholds. Auto clustering remains disabled, and no upper Power threshold is applied.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j98_kb_limit_foreground/run_20260908T030244Z/round_01/model_decision.json`

### run_20260908T030244Z / round_02

`action=None`, `decision_type=request_input`, `confidence=0.98`

```text
No safe autonomous forward action preserves the curated J99 particle output. Please expose an extraction candidate whose particles input connects to J99 output particles, while micrographs connect to J99 output micrographs.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j98_kb_limit_foreground/run_20260908T030244Z/round_02/model_decision.json`

### run_20260908T030244Z / round_03

`action=None`, `decision_type=request_input`, `confidence=0.99`

```text
A corrected extraction action is required to preserve the curated picks. Expose Extract From Micrographs with both micrographs and particles connected to J99 outputs.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j98_kb_limit_foreground/run_20260908T030244Z/round_03/model_decision.json`

### run_20260908T030244Z / round_04

`action=None`, `decision_type=request_input`, `confidence=0.99`

```text
A corrected extraction candidate is required to preserve the curated J99 particle set. Routing extraction from J98 would restore 100629 rejected picks.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j98_kb_limit_foreground/run_20260908T030244Z/round_04/model_decision.json`

## Extraction recovery

### run_20260908T031718Z / round_01

`action=extract_micrographs_multi`, `decision_type=forward`, `confidence=0.86`

```text
Proceed with GPU particle extraction from the completed Inspect Picks outputs. A 320-pixel source box spans about 274 A, providing safer padding around the approximately 196 A particle than the 256-pixel default; Fourier cropping to 256 pixels yields about 1.07 A/pixel and remains adequate for the reported 3.96 A target while reducing downstream computation.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j99_connection_fix_foreground/run_20260908T031718Z/round_01/model_decision.json`

## Box-size/Class 2D recovery

### run_20260908T052957Z / round_01

`action=extract_micrographs_multi`, `decision_type=forward`, `confidence=0.98`

```text
Submit both mandatory box-size extraction trials together using the curated J99 picks; all downstream actions remain blocked until the 384-pixel and 440-pixel trials complete.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j100_checkpoint_fix/run_20260908T052957Z/round_01/model_decision.json`

### run_20260908T052957Z / round_02

`action=extract_micrographs_multi`, `decision_type=forward`, `confidence=0.99`

```text
Retry the mandatory box-size sweep as two explicitly selected extraction jobs in one forward decision, assigning a separate integer box_size_pix to each trial and preserving the required J99 input connections.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j100_checkpoint_fix/run_20260908T052957Z/round_02/model_decision.json`

### run_20260908T052957Z / round_03

`action=class_2D_new`, `decision_type=branch`, `confidence=0.96`

```text
Begin the mandatory box-size comparison with the first completed extraction trial, J100. Its particles output is explicitly compatible with the available Class 2D action; defaults should be preserved so subsequent J101 and J102 trials can be evaluated under identical settings.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j100_checkpoint_fix/run_20260908T052957Z/round_03/model_decision.json`

### run_20260908T052957Z / round_04

`action=class_2D_new`, `decision_type=forward`, `confidence=0.98`

```text
Run the first mandatory Class 2D comparison on J100 particles. Starting with the 320-pixel trial and retaining defaults establishes a consistent baseline for later comparison with the 384- and 440-pixel extraction trials.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j100_checkpoint_fix/run_20260908T052957Z/round_04/model_decision.json`

### run_20260908T052957Z / round_07

`action=select_2D`, `decision_type=branch`, `confidence=0.91`

```text
Choose the J103 box-size branch because its contact sheet provides the best combination of sharp intact boundaries, lower noise, and diverse receptor orientations. Proceed with its supplied resolution-ranked 2D class selection before 3D initialization.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j100_checkpoint_fix/run_20260908T052957Z/round_07/model_decision.json`

## Final recovery from J105

### run_20260908T101054Z / round_01

`action=select_2D`, `decision_type=forward`, `confidence=0.93`

```text
Continue with the strongest completed box-size branch. J103 provides the best balance of particle detail, intact boundaries, low blur, and view diversity. There is no visual evidence of clipping that would justify fallback re-extraction, and the explicit selection removes the visually obvious class-33 artifact missed by resolution-only ranking.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_01/model_decision.json`

### run_20260908T101054Z / round_02

`action=homo_abinit`, `decision_type=forward`, `confidence=0.91`

```text
Generate initial 3D references directly from the selected J106 particles. Multi-class ab-initio reconstruction is preferable to repeating 2D classification and can separate residual junk or structural heterogeneity; direct homogeneous refinement is currently blocked because no valid volume exists.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_02/model_decision.json`

### run_20260908T101054Z / round_03

`action=nonuniform_refine_new`, `decision_type=forward`, `confidence=0.82`

```text
Proceed from the completed multi-class ab-initio result to the preferred current-generation non-uniform refinement using the registry-validated aggregate particle and volume outputs. C1 defaults match the dataset symmetry.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_03/model_decision.json`

### run_20260908T101054Z / round_04

`action=homo_refine_new`, `decision_type=forward`, `confidence=0.84`

```text
Run current-generation homogeneous refinement on the largest matched ab-initio class. Using particles_class_1 with volume_class_1 preserves class correspondence and avoids the confirmed multi-volume input type mismatch.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_04/model_decision.json`

### run_20260908T101054Z / round_05

`action=nonuniform_refine_new`, `decision_type=forward`, `confidence=0.93`

```text
Run modern non-uniform refinement from the completed homogeneous refinement outputs to improve reconstruction quality while preserving the established C1 solution and using all accepted particles.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_05/model_decision.json`

### run_20260908T101054Z / round_06

`action=validation`, `decision_type=forward`, `confidence=0.9`

```text
Validate the completed homogeneous refinement half-maps before further refinement. This is a preferred live next stage, requires only J109's compatible volume output, and avoids repeating the currently disallowed failed non-uniform refinement action while another non-uniform branch is already building.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_06/model_decision.json`

### run_20260908T101054Z / round_07

`action=nonuniform_refine_new`, `decision_type=forward`, `confidence=0.89`

```text
Run modern non-uniform refinement on the completed J109 particle alignments and volume, using the validated J110 mask. This is the most appropriate available refinement step for a heterogeneous membrane-protein complex and may close the gap from 4.57 A toward the reported 3.96 A result.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_07/model_decision.json`

### run_20260908T101054Z / round_08

`action=validation`, `decision_type=forward`, `confidence=0.96`

```text
Validate the completed J111 non-uniform refinement map using its compatible refinement mask before any additional optimization or postprocessing.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_08/model_decision.json`

### run_20260908T101054Z / round_09

`action=ctf_refine_global`, `decision_type=forward`, `confidence=0.82`

```text
Apply global CTF corrections to the validated consensus refinement before another reconstruction/refinement cycle; this is a meaningful optimization step toward improving the current 4.23 A map.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_09/model_decision.json`

### run_20260908T101054Z / round_10

`action=nonuniform_refine_new`, `decision_type=forward`, `confidence=0.92`

```text
Rerun current Non-uniform Refinement using the globally CTF-refined particles while retaining the established reference volume and validated mask.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_10/model_decision.json`

### run_20260908T101054Z / round_11

`action=validation`, `decision_type=forward`, `confidence=0.94`

```text
Run independent FSC validation on the post-global-CTF-refinement J114 map to quantify the updated reconstruction before any further refinement or postprocessing.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_11/model_decision.json`

### run_20260908T101054Z / round_12

`action=ctf_refine_local`, `decision_type=forward`, `confidence=0.82`

```text
Apply per-particle defocus refinement after the completed global CTF refinement, using the validated J115 half maps and mask with J114 particles. This is a meaningful optimization before another final 3D refinement and avoids simply repeating non-uniform refinement with unchanged CTF parameters.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_12/model_decision.json`

### run_20260908T101054Z / round_13

`action=nonuniform_refine_new`, `decision_type=forward`, `confidence=0.94`

```text
Run a fresh Non-uniform Refinement to apply the locally refined CTF parameters from J116, using the established J114 reference volume and J115 validation mask.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_13/model_decision.json`

### run_20260908T101054Z / round_14

`action=validation`, `decision_type=forward`, `confidence=0.98`

```text
Run FSC validation on the latest post-local-CTF-refinement map using the established compatible mask from J115 before attempting additional refinement or postprocessing.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_14/model_decision.json`

### run_20260908T101054Z / round_15

`action=None`, `decision_type=stop`, `confidence=0.91`

```text
The standard refinement and CTF-refinement cycle is complete and validated. With no configured target requiring continuation or failure requiring recovery, another refinement using the same J117 particles and volume would be an unsupported redundant iteration.
```

Source: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/round_15/model_decision.json`

## Final Run Result

The final recovery run ended with `stop_reason=model_stop` after 15 rounds. The last validated result was approximately 4.296 A FSC resolution at `J118`.

Primary final-run log directory: `/home/lisongyang/cryoagent/logs/closed_model_autonomous_empiar11280_w8_resume_after_j105_prompt_fix/run_20260908T101054Z/`
