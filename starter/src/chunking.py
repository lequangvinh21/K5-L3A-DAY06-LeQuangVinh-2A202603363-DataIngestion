"""Fixed-size and paragraph-aware chunking with source offsets."""
from hashlib import sha256
import re

from .models import Chunk, DocumentPage


def _chunk(
    page: DocumentPage,
    strategy: str,
    start: int,
    end: int,
) -> Chunk:
    text = page.text[start:end].strip()
    leading = len(page.text[start:end]) - len(page.text[start:end].lstrip())
    trailing = len(page.text[start:end]) - len(page.text[start:end].rstrip())
    source_start = start + leading
    source_end = end - trailing
    digest = sha256(text.encode("utf-8")).hexdigest()[:12]
    metadata = dict(page.metadata)
    metadata.update(
        {
            "strategy": strategy,
            "start_char": source_start,
            "end_char": source_end,
        }
    )
    return Chunk(
        chunk_id=f"{page.document_id}-p{page.page:03d}-{strategy}-{source_start}-{digest}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(
    page: DocumentPage,
    size: int,
    overlap: int,
    strategy: str = "fixed",
) -> list[Chunk]:
    """Split a page into overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not strategy:
        raise ValueError("strategy must not be empty")

    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_chunk(page, strategy, start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n+", text):
        left, right = start, separator.start()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = separator.end()

    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def _split_oversized_span(text: str, start: int, end: int, max_size: int) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\S+", text[start:end]))
    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = start

    for word in words:
        word_start = start + word.start()
        word_end = start + word.end()
        if word_end - word_start > max_size:
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
            for offset in range(word_start, word_end, max_size):
                spans.append((offset, min(offset + max_size, word_end)))
            continue

        if current_start is None:
            current_start, current_end = word_start, word_end
        elif word_end - current_start <= max_size:
            current_end = word_end
        else:
            spans.append((current_start, current_end))
            current_start, current_end = word_start, word_end

    if current_start is not None:
        spans.append((current_start, current_end))
    return spans


def by_paragraph(
    page: DocumentPage,
    max_size: int,
    strategy: str = "paragraph",
) -> list[Chunk]:
    """Group source paragraphs up to max_size and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not strategy:
        raise ValueError("strategy must not be empty")

    spans: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(page.text):
        spans.extend(_split_oversized_span(page.text, start, end, max_size))

    grouped: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    for start, end in spans:
        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= max_size:
            current_end = end
        else:
            grouped.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        grouped.append((current_start, current_end))

    return [
        _chunk(page, strategy, start, end)
        for start, end in grouped
        if page.text[start:end].strip()
    ]
