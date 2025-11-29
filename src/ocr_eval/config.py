"""Configuration loader for OCR Eval.

Reads settings from environment variables (optionally via .env) with sane defaults
so engines/CLI don't need to repeat os.getenv logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_profile: str = os.getenv("AWS_PROFILE", "textract-profile")
    temp_dir: str = os.getenv("OCR_EVAL_TEMP_DIR", "/tmp/ocr_eval_images")


def get_settings() -> Settings:
    return Settings()


# Central registry of datasets we support and their default splits.
DATASET_CONFIG = {
    "docvqa": {
        "hf_id": "pixparse/docvqa-wds",
        "default_split": "train",
        "trust_remote_code": False,
        "title": "DocVQA",
    },
    "funsd": {
        "hf_id": "nielsr/funsd",
        "default_split": "train",
        "trust_remote_code": False,
        "title": "FUNSD",
    },
    "cord": {
        "hf_id": "naver-clova-ix/cord-v2",
        "default_split": "test",
        "trust_remote_code": True,
        "title": "CORD",
    },
}
