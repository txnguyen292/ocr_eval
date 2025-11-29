"""Preview helpers for DocVQA and FUNSD datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from PIL import Image

from ..config import DATASET_CONFIG

__all__ = ["preview_docvqa_sample", "preview_funsd_sample"]

_IMAGE_KEYS: tuple[str, ...] = ("image", "png", "image_path", "image_file", "file_name")


def _ensure_image(example: Dict[str, Any], images_root: Optional[Path] = None) -> Image.Image:
    for key in _IMAGE_KEYS:
        if key not in example:
            continue
        value = example[key]
        if value is None:
            continue
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            if images_root and not path.is_absolute():
                path = images_root / path
            if not path.exists():
                continue
            return Image.open(path)
    raise ValueError("Could not resolve an image for the given example")


def _resolve_field(example: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key not in example:
            continue
        value = example[key]
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                if "question" in first:
                    return first["question"]
                if "text" in first:
                    return first["text"]
        if isinstance(value, dict):
            for nested_key in ("question", "answer", "text"):
                if nested_key in value and isinstance(value[nested_key], str):
                    return value[nested_key]
    return None


def _pick_sample(dataset: Dataset, sample_idx: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    if sample_idx is not None:
        return dataset[int(sample_idx)]
    return dataset.shuffle(seed=seed).select(range(1))[0]


def _show(image: Image.Image, title: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def preview_docvqa_sample(
    dataset_name: str = "pixparse/docvqa-wds",
    split: str = "validation",
    sample_idx: Optional[int] = None,
    seed: int = 42,
    *,
    show: bool = True,
    images_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    ds = load_dataset(dataset_name, split=split)
    example = _pick_sample(ds, sample_idx=sample_idx, seed=seed)
    image = _ensure_image(example, Path(images_root) if images_root else None)
    meta = example.get("json", {})
    question = meta.get("question") or example.get("question") or _resolve_field(example, ("question", "questions"))
    answers_raw = meta.get("answers") or example.get("answers") or _resolve_field(example, ("answer", "answers"))
    answer = answers_raw
    if isinstance(answers_raw, str) and answers_raw.startswith("[") and "]" in answers_raw:
        try:
            import ast
            parsed = ast.literal_eval(answers_raw)
            if isinstance(parsed, list) and parsed:
                answer = parsed[0]
        except Exception:
            pass

    if show:
        cfg = next(c for c in DATASET_CONFIG.values() if c.get("hf_id") == dataset_name)
        _show(image, cfg["title"])
        if question:
            print("Question:", question)
        if answer:
            print("Answer:", answer)

    result = dict(example)
    result["question_text"] = question
    result["answer_text"] = answer
    return result


def preview_funsd_sample(
    dataset_name: str = "nielsr/funsd",
    split: str = "train",
    sample_idx: Optional[int] = None,
    seed: int = 7,
    *,
    show: bool = True,
    images_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    ds = load_dataset(dataset_name, split=split)
    example = _pick_sample(ds, sample_idx=sample_idx, seed=seed)
    image = _ensure_image(example, Path(images_root) if images_root else None)
    question = _resolve_field(example, ("question", "questions", "text"))
    answer = _resolve_field(example, ("answer", "answers", "label"))

    if show:
        cfg = next(c for c in DATASET_CONFIG.values() if c.get("hf_id") == dataset_name)
        _show(image, cfg["title"])
        if question:
            print("Question:", question)
        if answer:
            print("Answer:", answer)

    result = dict(example)
    result["question_text"] = question
    result["answer_text"] = answer
    return result
