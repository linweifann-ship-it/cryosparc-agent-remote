# Minimal V3 Error Analysis Report

## Scope

This report analyzes the evaluation results of the `minimal_v3` model:

- adapter: `/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v3-minschema-from-taskcpt`
- evaluation summary: `/home/lisongyang/cryoagent/logs/eval_decision_accuracy_1240697_minv3_rerun/summary.json`
- per-sample results: `/home/lisongyang/cryoagent/logs/eval_decision_accuracy_1240697_minv3_rerun/per_sample_results.jsonl`

The `minimal_v3` schema predicts only:

- `decision_type`
- `selected_actions[].job_type`
- `selected_actions[].parameters`

## Overall Metrics

- `sample_count = 148`
- `valid_json_rate = 0.9932`
- `decision_type_accuracy = 0.9257`
- `selected_action_count_accuracy = 0.9189`
- `selected_action_set_accuracy = 0.8243`
- `selected_action_parameters_accuracy = 0.4865`
- `core_decision_exact_match_accuracy = 0.4797`

## Decision-Type Breakdown

### Branch

- `count = 30`
- `decision_type_accuracy = 0.9000`
- `selected_action_count_accuracy = 0.8333`
- `selected_action_set_accuracy = 0.8333`
- `selected_action_parameters_accuracy = 0.1333`
- `core_decision_exact_match_accuracy = 0.1333`

### Forward

- `count = 98`
- `decision_type_accuracy = 0.9388`
- `selected_action_count_accuracy = 0.9388`
- `selected_action_set_accuracy = 0.7959`
- `selected_action_parameters_accuracy = 0.5000`
- `core_decision_exact_match_accuracy = 0.5000`

### Stop

- `count = 19`
- `decision_type_accuracy = 0.9474`
- `selected_action_count_accuracy = 1.0000`
- `selected_action_set_accuracy = 1.0000`
- `selected_action_parameters_accuracy = 1.0000`
- `core_decision_exact_match_accuracy = 0.9474`

## Branch-Specific Findings

### Main Conclusion

`minimal_v3` substantially fixes the previous branch action-selection problem.
The model now usually chooses the correct set of parallel job types, but still struggles to choose the exact parameter values for those jobs.

### Branch Subsets

#### All branch samples

- `count = 30`
- `selected_action_set_accuracy = 0.8333`
- `selected_action_parameters_accuracy = 0.1333`

#### Non-`m` branch samples

- `count = 28`
- `decision_type_accuracy = 0.9643`
- `selected_action_count_accuracy = 0.8929`
- `selected_action_set_accuracy = 0.8929`
- `selected_action_parameters_accuracy = 0.1429`

#### `m`-variant branch samples

- `count = 2`
- all tracked accuracies are `0.0`

### Interpretation

The branch decision structure is now mostly correct for normal samples.
The remaining branch weakness is parameter grounding, not branch topology.
The two `m`-variant branch failures remain hard cases and should be inspected separately.

## Parameter Error Buckets

Parameter mismatches were grouped into semantic buckets.
Counts below refer to mismatched fields, not samples.

### Global mismatch counts

- `refine_switches = 37`
- `acquisition_physics = 29`
- `path_like = 21`
- `resource_compute = 18`
- `picker_extract_geometry = 10`
- `other = 5`
- `import_flags = 1`

### Branch-only mismatch counts

- `acquisition_physics = 27`
- `path_like = 20`
- `resource_compute = 5`
- `picker_extract_geometry = 2`
- `other = 1`
- `import_flags = 1`
- `refine_switches = 0`

### Forward-only mismatch counts

- `refine_switches = 37`
- `resource_compute = 13`
- `picker_extract_geometry = 8`
- `other = 4`
- `path_like = 1`
- `acquisition_physics = 2`

## Most Frequent Mismatched Fields

### Top global fields

- `psize_A = 14`
- `blob_paths = 12`
- `total_dose_e_per_A2 = 11`
- `compute_num_gpus = 10`
- `box_size_pix = 8`
- `refine_ctf_global_refine = 6`
- `refine_defocus_refine = 6`
- `particle_meta_path = 5`
- `particle_blob_path = 4`
- `refine_init_shift = 4`
- `refine_init_twist = 4`

### Branch-only top fields

- `psize_A = 13`
- `blob_paths = 11`
- `total_dose_e_per_A2 = 10`
- `particle_meta_path = 5`
- `compute_num_gpus = 5`
- `particle_blob_path = 4`
- `accel_kv = 2`
- `cs_mm = 2`
- `n_templates = 2`

## Most Frequent Error-Prone Job Types

### Global top job types

- `homo_refine_new = 9`
- `extract_micrographs_multi = 8`
- `import_micrographs = 6`
- `import_movies = 6`
- `import_particles = 5`
- `patch_ctf_estimation_multi = 5`
- `helix_refine = 4`
- `filament_tracer_gpu = 3`

### Branch-only top job types

- `import_micrographs = 6`
- `import_particles = 5`
- `import_movies = 5`
- `patch_ctf_estimation_multi = 3`
- `create_templates = 2`
- `patch_motion_correction_multi = 2`

## Interpretation of Error Types

### 1. Branch parameter errors are dominated by dataset-specific grounding

For branch samples, the largest error sources are:

- path-like parameters
- acquisition / microscope parameters
- a smaller number of resource choices

This means the model usually knows which branch jobs to run, but still substitutes template-like values from similar datasets.

Typical branch error fields:

- `blob_paths`
- `particle_blob_path`
- `particle_meta_path`
- `psize_A`
- `total_dose_e_per_A2`
- `compute_num_gpus`

### 2. Forward parameter errors are dominated by refine switches and extraction settings

For forward samples, the biggest error bucket is `refine_switches`.
That suggests the model correctly identifies the refine job type, but still misses detailed boolean or mode configuration.

Typical forward error fields:

- `refine_ctf_global_refine`
- `refine_defocus_refine`
- `crg_do_spherical`
- `crg_do_anisomag`
- `crg_do_tetrafoil`
- `refine_init_shift`
- `refine_init_twist`

### 3. Resource defaults are still over-regularized

The model often outputs common default values for:

- `compute_num_gpus`
- `box_size_pix`
- `diameter`

This suggests the model has learned common operational priors but not always the dataset-specific target configuration.

## Representative Error Examples

### Branch: import particles path + acquisition drift

Sample: `EMPIAR-10059:v2_step:000`

- wrong fields:
  - `particle_meta_path`
  - `alignments3D_exists`
  - `cs_mm`
  - `psize_A`

### Branch: import micrographs path + dose drift

Sample: `EMPIAR-10192:v2_step:000`

- wrong fields:
  - `blob_paths`
  - `psize_A`
  - `total_dose_e_per_A2`

### Branch: patch CTF GPU count drift

Sample: `EMPIAR-10192:v2_step:001`

- wrong field:
  - `compute_num_gpus`

### Forward: template picker diameter drift

Sample: `EMPIAR-10192:v2_step:002`

- wrong field:
  - `diameter`

### Forward: extract micrographs box size drift

Sample: `EMPIAR-10192:v2_step:004`

- wrong field:
  - `box_size_pix`

### Forward: refine switch omission

Sample: `EMPIAR-10202:v2_step:002`

- wrong fields:
  - `crg_do_spherical`
  - `crg_do_anisomag`
  - `crg_do_tetrafoil`
  - `refine_do_ews_correct`

## Practical Takeaways

### What is already solved well

- strict JSON output is stable
- action-set prediction is much better than the old full-schema setup
- branch action selection is now mostly correct for non-`m` samples

### What still limits accuracy

- dataset-specific path parameters
- acquisition / microscope numeric settings
- detailed refine toggles
- resource defaults that drift to common values

### Suggested next step

The next optimization target should be parameter grounding, not schema redesign.
A good next analysis step would be to augment the model input with more explicit parameter-bearing fields from:

- import job state
- extraction job state
- refine job state
- dataset-level acquisition metadata

That should be especially helpful for the top-mismatch fields:

- `psize_A`
- `blob_paths`
- `total_dose_e_per_A2`
- `compute_num_gpus`
- `box_size_pix`
- refine-switch flags
