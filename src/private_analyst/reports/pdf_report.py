from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from private_analyst.reports.exporters import report_markdown
from private_analyst.analytics.operations import OperationResult


def to_pdf_bytes(result: OperationResult, explanation: str, include_explanation: bool) -> bytes:
    markdown = report_markdown(result, explanation, include_explanation)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Private Analyst AI report")
    styles = getSampleStyleSheet()
    story = []
    for line in markdown.splitlines():
        text = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(text or " ", styles["Normal"]))
        story.append(Spacer(1, 4))
    doc.build(story)
    return buffer.getvalue()
