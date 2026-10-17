from __future__ import annotations

from io import BytesIO

from reportlab.pdfgen import canvas

from private_analyst.config import Limits
from private_analyst.documents.index import LocalTextIndex
from private_analyst.documents.pdf import extract_pdf_text


def _pdf_with_text(text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(72, 720, text)
    c.save()
    return buffer.getvalue()


def test_pdf_extraction_and_search() -> None:
    limits = Limits()
    data = _pdf_with_text("East warehouse Widget demand in January.")
    extract = extract_pdf_text(data, "note.pdf", limits)
    assert extract.pages[0].page_number == 1
    assert "Widget" in extract.pages[0].text
    index = LocalTextIndex()
    index.add_extract(extract)
    index.add_plain_text("other.txt", "Unrelated cooking recipe.")
    hits = index.search("Widget demand warehouse")
    assert hits
    assert hits[0].document_name == "note.pdf"
    assert hits[0].page_number == 1


def test_delete_and_rebuild() -> None:
    index = LocalTextIndex()
    index.add_plain_text("keep.txt", "alpha unique token zebra")
    index.add_plain_text("drop.txt", "beta unique token yak")
    assert index.delete_document("drop.txt") == 1
    index.rebuild()
    hits = index.search("yak")
    assert all(h.document_name != "drop.txt" for h in hits)
