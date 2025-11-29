## OCR Eval Toolkit

Minimal helpers to compare OCR engines (Textract, VLM) on public benchmarks.

### Datasets
- **DocVQA**: `pixparse/docvqa-wds`
- **FUNSD**: `nielsr/funsd`
- **CORD**: `naver-clova-ix/cord-v2`

### CLI
Run an evaluation with your chosen dataset and engines:
```bash
pip install -e .
python -m ocr_eval.cli evaluate --dataset docvqa --engine all --samples 10
```

### Notebooks
- `notebooks/docvqa.ipynb` and `notebooks/funsd.ipynb` preview samples via `ocr_eval.utils`.
