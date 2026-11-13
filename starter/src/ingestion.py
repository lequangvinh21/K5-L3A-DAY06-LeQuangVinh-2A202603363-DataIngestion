"""Read PDF pages while preserving physical page numbers and useful structure."""
from pathlib import Path
import re
import unicodedata
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage


# Physical PDF page numbers where chapter introductions begin, from the table of contents.
_CHAPTER_START_PAGES = (13, 23, 37, 47, 57, 67, 79, 87, 97, 111, 119, 129, 135, 145)


def chapter_for_page(page_number: int) -> int | None:
    """Return the chapter number for a physical page in this source PDF."""
    for index, start_page in enumerate(_CHAPTER_START_PAGES):
        next_start = (
            _CHAPTER_START_PAGES[index + 1]
            if index + 1 < len(_CHAPTER_START_PAGES)
            else 154
        )
        if start_page <= page_number < next_start:
            return index + 1
    return None


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining paragraph breaks."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page and retain extraction warnings in its metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not document_id.strip():
        raise ValueError("document_id must not be empty")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        metadata: dict[str, Any] = {
            "page_id": f"{document_id}-p{page_number:03d}",
        }
        chapter = chapter_for_page(page_number)
        if chapter is not None:
            metadata["chapter"] = chapter
        if not text:
            metadata["extraction_warning"] = "No extractable text on this physical PDF page."

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )
    return pages
