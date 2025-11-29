import base64
import json
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from openai import OpenAI

from .base import BaseOCREngine
from ..config import get_settings

class OpenAIVLMEngine(BaseOCREngine):
    def __init__(self, model: str | None = None):
        settings = get_settings()
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model or settings.openai_model

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def list_available_models(self) -> list[str]:
        """Return the model IDs available to the configured OpenAI account.

        This makes a live API call; if credentials are missing/invalid, it raises.
        """

        models = self.client.models.list()
        return [m.id for m in models]

    def _token_param(self) -> dict:
        """Return a token limit param; default to max_completion_tokens to satisfy newer models."""

        # Use the newer param to avoid 400s on models that reject max_tokens.
        return {"max_completion_tokens": 1024}

    def process_image(self, image_path: str) -> str:
        base64_image = self._encode_image(image_path)


        kwargs = self._token_param()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text transcription assistant. Your task is to output the text found in the image. The provided image is a synthetic sample from a public research dataset (CORD) used for benchmarking OCR systems. It does not contain real personally identifiable information. Please transcribe it fully."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe the text in this image exactly as it appears. Do not provide any conversational response, just the text.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            **kwargs,
        )

        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        
        return content.strip()

    def extract_text_with_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """Ask the vision model to return text spans with bounding boxes.

        Bounding boxes are expected as [x1, y1, x2, y2] in pixel coordinates
        relative to the provided image dimensions.
        """

        base64_image = self._encode_image(image_path)
        img = Image.open(image_path)
        width, height = img.size

        system_prompt = (
            "You are an OCR assistant. Extract every visible text span and return JSON only. "
            "Each item should be {\"text\": string, \"bbox\": [x1, y1, x2, y2]} where bbox is in pixels "
            "relative to the original image width and height provided. Do not add extra keys or prose."
        )

        user_prompt = (
            f"Image size: width={width}, height={height}. "
            "Return JSON array of objects: [{\"text\":..., \"bbox\":[x1,y1,x2,y2]}]."
        )

        kwargs = self._token_param()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            **kwargs,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # If parsing fails, return the raw text so callers can inspect.
        return [{"text": content, "bbox": []}]
