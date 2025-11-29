from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets import load_dataset

from ..config import DATASET_CONFIG, get_settings

settings = get_settings()
TMP_DIR = Path(settings.temp_dir)
SUPPORTED_DATASETS = tuple(DATASET_CONFIG.keys())


def _save_image(image, stem: str) -> str:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    image_path = TMP_DIR / f"{stem}.png"
    image.save(image_path)
    return str(image_path)


def _extract_docvqa_text(example: Dict) -> str:
    """Join OCR lines from the DocVQA WDS sample."""
    try:
        rec = example["json"]["ocr_results"]["recognitionResults"][0]
        lines = rec.get("lines", [])
        return "\n".join(line.get("text", "") for line in lines if line.get("text"))
    except Exception:
        return ""


def _extract_funsd_text(example: Dict) -> str:
    """Join tokenized words from FUNSD into a flat string."""
    words = example.get("words") or []
    return " ".join(words)


def _extract_cord_text(example: Dict) -> str:
    """Extract plain text from CORD JSON ground truth."""
    text = ""
    try:
        gt_data = json.loads(example["ground_truth"])

        def extract_values(obj):
            values = []
            if isinstance(obj, dict):
                for v in obj.values():
                    values.extend(extract_values(v))
            elif isinstance(obj, list):
                for v in obj:
                    values.extend(extract_values(v))
            elif isinstance(obj, str):
                values.append(obj)
            elif isinstance(obj, (int, float)):
                values.append(str(obj))
            return values

        if "gt_parse" in gt_data:
            all_values = extract_values(gt_data["gt_parse"])
            text = "\n".join(all_values)
        else:
            text = str(gt_data)
    except Exception:
        text = ""
    return text.strip()


def load_dataset_samples(
    name: str = "docvqa",
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> List[Dict]:
    """Load a supported dataset and return a list of samples with image + text."""

    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {SUPPORTED_DATASETS}")

    cfg = DATASET_CONFIG[name]
    ds = load_dataset(cfg["hf_id"], split=split or cfg["default_split"], trust_remote_code=cfg.get("trust_remote_code", False))
    if num_samples:
        ds = ds.select(range(num_samples))

    # Dispatch to the appropriate sample builder.
    SAMPLE_BUILDERS: Dict[str, Callable[[Dict, int], Dict]] = {
        "docvqa": _build_docvqa_sample,
        "funsd": _build_funsd_sample,
        "cord": _build_cord_sample,
    }

    builder = SAMPLE_BUILDERS[name]
    return [builder(item, i) for i, item in enumerate(ds)]


def _build_docvqa_sample(item: Dict, idx: int) -> Dict:
    image = item.get("png") or item.get("image")
    text = _extract_docvqa_text(item)
    meta = item.get("json", {})
    question = meta.get("question") or ""
    answer = meta.get("answers") or ""
    return {
        "id": str(meta.get("questionId", idx)),
        "image_path": _save_image(image, f"docvqa_{idx}"),
        "ground_truth": text,
        "question": question,
        "answer": answer,
    }


def _build_funsd_sample(item: Dict, idx: int) -> Dict:
    image = item["image"]
    text = _extract_funsd_text(item)
    return {
        "id": item.get("id", str(idx)),
        "image_path": _save_image(image, f"funsd_{idx}"),
        "ground_truth": text,
    }


def _build_cord_sample(item: Dict, idx: int) -> Dict:
    image = item["image"]
    text = _extract_cord_text(item)
    return {
        "id": str(idx),
        "image_path": _save_image(image, f"cord_{idx}"),
        "ground_truth": text,
    }
