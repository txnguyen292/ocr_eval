# Progress and Next Steps

## What I did
1. **Generalized dataset loader**: Added support for DocVQA (`pixparse/docvqa-wds`) and FUNSD (`nielsr/funsd`), while keeping CORD. Text extraction is tailored per dataset (DocVQA OCR lines, FUNSD word tokens, CORD ground-truth JSON) and images are saved to `/tmp/ocr_eval_images`.
2. **CLI dataset selection**: Updated `ocr_eval.cli` to accept `--dataset` and `--split`, routing through the generalized loader so you can run Textract/VLM on DocVQA, FUNSD, or CORD.
3. **Package layout**: Standardized the `src/` structure for the `ocr_eval` package and refreshed notebook imports to use `ocr_eval.utils`.
4. **Docs/README refresh**: Documented supported datasets and a sample CLI invocation.
5. **OpenAI VLM path wired**: Implemented `OpenAIVLMEngine` (base64 → `gpt-4o` vision call) and hooked it into the CLI so `--engine openai`/`all` runs OCR via OpenAI Vision and reports WER/CER.

## What still needs doing
1. **Question/answer extraction for DocVQA**: The WDS sample stores questions under `json['question']` and answers under `json['answers']`; wire this into the loader/preview helpers so QA comparisons use ground-truth Q/A instead of inferred OCR text only.
2. **Evaluation metrics expansion**: Beyond CER/WER, add EM/F1 QA scoring and cost/latency logging to the CLI pipeline.
3. **Textract validation**: Add upfront checks for AWS creds/region and make failures clearer before running.
4. **Configurable output paths**: Let users choose where temp images/results go (not just `/tmp/ocr_eval_images`).
5. **Tests**: Add minimal tests for loaders (DocVQA/FUNSD) and metric helpers to catch schema drift.
6. **Boxed OCR comparison**: Add Textract line-level bbox extraction and a notebook that overlays Textract vs OpenAI VLM boxes on the same DocVQA image.
7. **BBox scoring**: Implement IoU-based box matching with detection metrics (precision/recall/F1) and text accuracy on matched pairs (exact match, CER/WER) to quantify VLM vs native DocVQA boxes instead of manual visual inspection.

Run `pip install -e .` before executing notebooks or the CLI to ensure imports resolve via the project root.

## Recreate the project from zero (step by step, no clone)
1. **Prereqs**: Python 3.12+, network access (HF datasets + OpenAI), optional AWS CLI for Textract.
2. **Scaffold and env**:
   ```
   mkdir ocr_eval && cd ocr_eval
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Create pyproject**: Add `pyproject.toml` with project metadata and deps: boto3, datasets, huggingface_hub, ipykernel, matplotlib, numpy, openai, pandas, Pillow, python-Levenshtein, rapidfuzz, sacrebleu, typer, python-dotenv; set `package-dir` to `src` and use setuptools backend.
4. **Add package skeleton** under `src/ocr_eval/`: `__init__.py`, `config.py` (env loader + `DATASET_CONFIG`), `cli.py` (Typer evaluate command), `data/loader.py` (DocVQA/FUNSD/CORD loaders saving temp images), `engines/{base.py,textract.py,openai.py}`, `utils/{metrics.py,preview.py}`, `legacy_utils.py`.
5. **Docs/notebooks**: Create `docs/progress_and_plan.md`, `docs/plan.json`, and `README.md` with dataset/CLI info; add `notebooks/` with `docvqa.ipynb`, `funsd.ipynb`, and optional bbox demos.
6. **Install editable**:
   ```
   pip install -e .
   ```
7. **Configure credentials**: Create `.env` (or export vars):
   ```
   OPENAI_API_KEY=...
   OPENAI_MODEL=gpt-4o
   AWS_PROFILE=textract-profile     # or use AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
   AWS_REGION=us-east-1
   OCR_EVAL_TEMP_DIR=/tmp/ocr_eval_images
   ```
8. **Smoke tests**:
   ```
   python -m ocr_eval.cli evaluate --dataset docvqa --engine openai --samples 2 --output results.md
   python -m ocr_eval.cli evaluate --dataset funsd --engine textract --samples 2 --output results.md
   ```
9. **Notebooks**: Open `notebooks/docvqa.ipynb` or `notebooks/funsd.ipynb` with the venv kernel.

## Project structure (key paths)
```
ocr_eval/
├── README.md
├── docs/
│   ├── plan.json
│   └── progress_and_plan.md
├── src/ocr_eval/
│   ├── cli.py                # CLI entrypoint
│   ├── config.py             # env/settings loader + dataset registry
│   ├── data/loader.py        # dataset loaders and temp image saving
│   ├── engines/              # OCR engines (Textract, OpenAI VLM)
│   ├── utils/                # metrics, preview helpers
│   └── legacy_utils.py
├── notebooks/                # exploration + bbox demos
├── main.py, inspect_cord.py
├── results.md
├── pyproject.toml
└── uv.lock
```
