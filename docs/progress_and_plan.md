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
