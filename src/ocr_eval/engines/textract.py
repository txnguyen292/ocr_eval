import boto3
from .base import BaseOCREngine
from ..config import get_settings

class TextractEngine(BaseOCREngine):
    def __init__(self, region_name: str | None = None):
        settings = get_settings()
        region = region_name or settings.aws_region
        # Prefer an explicit profile if provided (defaults to textract-profile in config).
        if settings.aws_profile:
            session = boto3.Session(profile_name=settings.aws_profile, region_name=region)
            self.client = session.client("textract")
        else:
            self.client = boto3.client("textract", region_name=region)

    def process_image(self, image_path: str) -> str:
        with open(image_path, "rb") as document:
            image_bytes = document.read()

        response = self.client.detect_document_text(Document={"Bytes": image_bytes})
        
        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text += item["Text"] + "\n"
        
        return text.strip()
