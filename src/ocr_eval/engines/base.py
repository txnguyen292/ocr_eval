from abc import ABC, abstractmethod

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def process_image(self, image_path: str) -> str:
        """
        Process an image and return the extracted text.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Extracted text from the image.
        """
        pass
