from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from private_analyst.config import Limits
from private_analyst.exceptions import ResourceLimitError, ValidationError


@dataclass
class PageText:
    page_number: int
    text: str
    extraction: str


@dataclass
class DocumentExtract:
    source_name: str
    pages: list[PageText] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[page {p.page_number}]\n{p.text}" for p in self.pages if p.text)


def extract_pdf_text(data: bytes, filename: str, limits: Limits) -> DocumentExtract:
    if len(data) > limits.max_upload_bytes:
        raise ResourceLimitError("PDF exceeds the configured size limit.")
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - fallback to installed PyPDF2
        from PyPDF2 import PdfReader  # type: ignore

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("The PDF could not be opened.") from exc

    pages: list[PageText] = []
    warnings: list[str] = []
    n_pages = len(reader.pages)
    if n_pages > limits.max_pdf_pages:
        raise ResourceLimitError(f"PDF page count exceeds the configured limit ({limits.max_pdf_pages}).")
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
            warnings.append(f"Text extraction failed on page {i}.")
        extraction = "text_layer" if text else "empty_text_layer"
        if not text:
            warnings.append(f"Page {i} has no extractable text layer. Local OCR may be required.")
        pages.append(PageText(page_number=i, text=text, extraction=extraction))
    return DocumentExtract(source_name=filename, pages=pages, warnings=warnings)
