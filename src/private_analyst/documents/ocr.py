from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRResult:
    text: str
    engine: str
    available: bool
    detail: str


def ocr_image(data: bytes) -> OCRResult:
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
    except ImportError:
        return OCRResult("", "tesseract", False, "Python OCR packages are not installed.")
    try:
        image = Image.open(BytesIO(data))
        text = pytesseract.image_to_string(image)
        return OCRResult(text.strip(), "tesseract", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return OCRResult("", "tesseract", False, type(exc).__name__)
