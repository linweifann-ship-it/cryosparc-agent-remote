# Task-Relevant Docs Layout

## Goal

This layout is for collecting local snapshots of:

- cryoSPARC docs
- cryoSPARC Live docs
- RELION tutorials
- cryoDRGN docs
- EMPIAR tutorials
- future task-relevant manuals and FAQ pages

These local files can then be normalized with:

- `prepare_task_relevant_docs_cpt_corpus.py`

## Recommended directory structure

```text
task_relevant_docs_raw/
  cryosparc_guide_core/
    jobs.md
    workflows.md
    all_job_types.md
  cryosparc_live_docs/
    live_start_to_finish.md
    live_faq.md
  relion_docs/
    single_particle_tutorial.md
    on_the_fly_processing.md
  cryodrgn_docs/
    README.md
    usage_examples.md
  empiar_tutorials/
    quick_tour.md
    faq.md
```

The top-level subdirectory name should match the `source_id` in:

- [task_relevant_cpt_sources.json](/home/lisongyang/cryoagent/task_relevant_cpt_sources.json)

## Supported file formats

- `.md`
- `.markdown`
- `.txt`
- `.html`
- `.htm`
- `.json`

## JSON snapshot format

If you store a page snapshot as JSON, this structure is recommended:

```json
{
  "title": "Jobs",
  "source_url": "https://guide.cryosparc.com/...",
  "text": "Page content here..."
}
```

Alternative keys also accepted by the current script:

- `content`
- `body`
- `markdown`
- `name`
- `page_title`
- `url`
- `homepage`

## Example command

```bash
python3 /home/lisongyang/cryoagent/prepare_task_relevant_docs_cpt_corpus.py \
  --docs-root /home/lisongyang/cryoagent/task_relevant_docs_raw \
  --output-root /home/lisongyang/cryoagent/task_relevant_docs_cpt \
  --source-manifest-json /home/lisongyang/cryoagent/task_relevant_cpt_sources.json \
  --write-cleaned-text
```

## Output files

The script writes:

- `docs_metadata.jsonl`
- `docs_cpt_chunks.jsonl`
- `failed_docs.jsonl`
- `manifest.json`

These outputs are designed to look similar to the current paper-based CPT outputs, so later mixing is straightforward.
