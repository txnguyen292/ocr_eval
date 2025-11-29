# Notes for Tung — DocVQA Evaluation Baseline

## Goal
Document why we should always keep DocVQA in the evaluation mix. Use it to stress-test both OCR+LLM and direct VLM pipelines before adopting any engine.

## Dataset Checklist
- Source: Hugging Face `donut-data/docvqa_task1` (official DocVQA Task 1).
- Download:
  ```bash
  mkdir -p data/docvqa_task1 && \
  huggingface-cli download donut-data/docvqa_task1 --repo-type dataset \
    --local-dir data/docvqa_task1
  ```
- Build manageable splits (~100 dev / 400 test):
  ```python
  from datasets import load_dataset
  ds = load_dataset("donut-data/docvqa_task1")
  ds["train"].select(range(100)).to_json("data/processed/questions/dev.jsonl")
  ds["validation"].select(range(400)).to_json("data/processed/questions/test.jsonl")
  ```
- Manifests: `data/processed/manifests/{dev,test}.jsonl` with `{page_id, image_path}` entries referencing the downloaded images.

## Evaluation Reminder
1. Long runs Textract → text LLM on these DocVQA splits (`--mode textract_llm`).
2. Huy runs VLM→QA (`--mode vlm_image`).
3. Compare EM/F1/latency/cost via `scripts/06_evaluate_qa.py`.

Keep this note so we always have DocVQA instructions handy without polluting Long/Huy briefs.
