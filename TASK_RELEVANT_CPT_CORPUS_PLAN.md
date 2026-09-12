# Task-Relevant CPT Corpus Plan

## Why expand beyond papers

The current CPT corpus is dominated by generic scientific paper text. That helps with:

- domain language fluency
- formatting stability
- general cryo-EM vocabulary

But it does **not** directly teach:

- how cryoSPARC jobs are sequenced
- what signals in logs imply the next action
- which parameters are typical defaults versus dataset-specific overrides
- how interactive and branching workflow decisions are made

For this project, CPT should move closer to **procedural EM operations knowledge**, not just scientific prose.

## Recommendation

Yes: we should add **cryoSPARC-related tutorials, official docs, workflow guides, troubleshooting notes, and tool manuals** as CPT data.

In fact, these sources are likely more valuable than generic papers for the downstream task.

## Priority order

### P0: cryoSPARC official docs and tutorials

These should be the highest-priority new CPT sources because the downstream output schema is currently centered on cryoSPARC jobs and parameters.

Suggested sections:

- CryoSPARC Guide
  - `Jobs`
  - `Creating and Running Jobs`
  - `Inspecting Job Data`
  - `Workflows`
  - `Managing Jobs`
  - `Interactive Jobs`
  - `All Job Types in CryoSPARC`
  - `Automated Workflows`
  - `Data Processing Tutorials`
- CryoSPARC Live docs
  - `New Live Session: Start to Finish Guide`
  - `Live Jobs and Session-Level Functions`
  - `Performance Metrics`
  - `FAQs and Troubleshooting`

Why it matters:

- directly teaches job names, workflow structure, and operational semantics
- much closer to the model's action space than papers
- likely improves parameter priors in the right direction

## P0.5: local workflow artifacts already in this project

These should remain central, even if they are technically not "CPT" in the traditional sense.

Relevant local sources:

- workflow JSONs
- workflow labels
- structured log summaries
- MCP interaction traces
- internal notes about live runs and error handling

Why it matters:

- they define the actual task distribution
- they contain the exact job names and parameter surfaces we need
- they align with the real deployment environment

## P1: cryo-EM tool manuals and tutorials for adjacent systems

These are useful both for better cryo-EM reasoning and for future multi-tool expansion.

Suggested sources:

- RELION official docs
  - single-particle tutorial
  - subtomogram tutorial
  - on-the-fly processing
  - reference pages
- cryoDRGN docs / README / tutorials
- cisTEM / EMAN2 / Scipion official manuals if future tool coverage matters

Why it matters:

- teaches general cryo-EM workflow concepts that transfer across tools
- helps the model reason about common pipeline stages even when no known workflow is provided
- useful if the assistant later expands beyond cryoSPARC

## P1.5: archive metadata and training resources

Suggested sources:

- EMPIAR archive pages and tutorials
- EMDB dataset descriptions
- EMPIAR / EMDB talks and tutorials
- dataset quick tours and FAQ material

Why it matters:

- improves understanding of dataset types and experimental context
- helps connect `empiar_id`, `emdb_id`, specimen type, and likely processing choices
- useful for upstream dataset-context interpretation

## P2: troubleshooting and decision-oriented notes

Suggested sources:

- official FAQs
- tutorial transcripts
- forum posts or curated Q&A summaries
- internal lab SOPs
- runbooks for common cryoSPARC failure/recovery patterns

Why it matters:

- many workflow decisions are triggered by failure states, quality issues, or incomplete outputs
- this is closer to agent behavior than polished papers are

## What to avoid or down-weight

These sources are not useless, but they should not dominate the CPT mix:

- generic structural biology papers with no operational workflow content
- non-cryo-EM scientific papers
- papers that mention cryo-EM only in passing
- long theory-heavy texts with little procedural content
- duplicated tutorial copies or low-quality OCR

## Proposed corpus buckets

Instead of one mixed CPT corpus, build labeled buckets:

1. `procedural_docs`
   - cryoSPARC docs
   - RELION docs
   - tool manuals

2. `workflow_examples`
   - workflow JSON-derived naturalized text
   - log summaries
   - MCP traces

3. `dataset_context`
   - EMDB XML abstracts
   - EMPIAR metadata
   - archive tutorials

4. `research_papers`
   - cryo-EM method papers
   - cryoSPARC / RELION / cryoDRGN papers
   - task-relevant experimental papers

This lets us control sampling weights rather than throwing everything into one pool.

## Suggested sampling weights for the next CPT round

For a task-oriented CPT experiment, a reasonable starting point is:

- `procedural_docs`: `35%`
- `workflow_examples`: `30%`
- `dataset_context`: `15%`
- `research_papers`: `20%`

This is only a starting point, but it is much better aligned than the current paper-heavy mix.

## Recommended filtering rules for papers

Keep papers if they match at least one strong relevance signal:

- mentions `cryoSPARC`, `RELION`, `cryoDRGN`, `single-particle`, `subtomogram`, `particle picking`, `motion correction`, `CTF`, `2D classification`, `3D refinement`, `ab initio reconstruction`
- focuses on cryo-EM processing methods
- describes practical reconstruction workflows
- discusses parameter sensitivity or failure modes

Down-rank or exclude if:

- no cryo-EM processing content
- only biochemical/structural interpretation with no workflow relevance
- unrelated physics/chemistry/math papers

## Best next experiment

The most informative next CPT experiment is:

1. Build a **filtered, task-relevant CPT corpus**
2. Run `filtered CPT -> SFT`
3. Compare against `SFT-only`
4. Focus on:
   - `selected_action_set_accuracy`
   - `selected_action_parameters_accuracy`
   - `core_decision_exact_match_accuracy`

If those do not improve, then task-relevant CPT is probably not the main bottleneck, and effort should shift to:

- stricter SFT target design
- better state representation
- more task-specific supervised data

## Good source links

These are good starting points for collection:

- CryoSPARC Guide: `https://guide.cryosparc.com/`
- RELION docs: `https://relion.readthedocs.io/en/latest/`
- EMPIAR: `https://www.ebi.ac.uk/empiar/`
- cryoDRGN repository: `https://github.com/ml-struct-bio/cryodrgn`

## Immediate action items

1. Build a source manifest for cryoSPARC / RELION / EMPIAR / cryoDRGN docs.
2. Write a relevance filter for the existing paper corpus.
3. Separate the CPT corpus into bucketed JSONL files.
4. Run a smaller filtered-CPT ablation before committing full compute again.
