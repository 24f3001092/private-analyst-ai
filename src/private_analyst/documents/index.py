"""Local lexical search. No remote vector database."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from private_analyst.documents.pdf import DocumentExtract


@dataclass
class Passage:
    document_name: str
    page_number: int | None
    text: str
    score: float


def _chunks(extract: DocumentExtract, size: int = 800) -> list[dict]:
    items: list[dict] = []
    for page in extract.pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        for i in range(0, len(text), size):
            piece = text[i : i + size]
            items.append(
                {
                    "document_name": extract.source_name,
                    "page_number": page.page_number,
                    "text": piece,
                }
            )
    return items


class LocalTextIndex:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add_extract(self, extract: DocumentExtract) -> int:
        added = _chunks(extract)
        self.records.extend(added)
        self._rebuild()
        return len(added)

    def add_plain_text(self, name: str, text: str) -> int:
        extract = DocumentExtract(source_name=name)
        from private_analyst.documents.pdf import PageText

        extract.pages = [PageText(page_number=1, text=text, extraction="plain")]
        return self.add_extract(extract)

    def delete_document(self, name: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r["document_name"] != name]
        self._rebuild()
        return before - len(self.records)

    def rebuild(self) -> None:
        self._rebuild()

    def search(self, query: str, k: int = 5) -> list[Passage]:
        query = query.strip()
        if not query or self._matrix is None or not self.records:
            return []
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        out: list[Passage] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            rec = self.records[int(idx)]
            out.append(
                Passage(
                    document_name=rec["document_name"],
                    page_number=rec.get("page_number"),
                    text=rec["text"],
                    score=round(score, 4),
                )
            )
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.records = json.loads(path.read_text(encoding="utf-8"))
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.records:
            self._vectorizer = None
            self._matrix = None
            return
        corpus = [re.sub(r"\s+", " ", r["text"]) for r in self.records]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def as_dicts(self) -> list[dict]:
        return list(self.records)
