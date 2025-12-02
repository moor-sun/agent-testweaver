# rag/loaders/pdf_loader.py
from typing import List
from pypdf import PdfReader  # pip install pypdf


def load_pdf_as_text(pdf_path: str) -> str:
    """
    Legacy helper: returns full text of a PDF as a single string.
    Still useful sometimes, but not ideal for RAG.
    """
    reader = PdfReader(pdf_path)
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n\n".join(chunks)


def _normalize_whitespace(text: str) -> str:
    """
    Collapse awkward line breaks and extra spaces.
    """
    # Replace Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Sometimes PDFs break lines mid-sentence, so join single newlines
    lines = text.split("\n")
    merged_lines: List[str] = []
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # blank line → paragraph break
            if buffer:
                merged_lines.append(" ".join(buffer))
                buffer = []
            merged_lines.append("")  # keep a blank as paragraph separator
        else:
            buffer.append(stripped)

    if buffer:
        merged_lines.append(" ".join(buffer))

    # Rebuild with explicit blank lines between paragraphs
    return "\n".join(merged_lines)


def load_pdf_as_chunks(
    pdf_path: str,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Load a PDF and return a list of text chunks suitable for RAG.

    Strategy:
    - Extract text from all pages
    - Normalize whitespace
    - Split into paragraphs on blank lines
    - Build overlapping windows of ~max_chars characters
    """

    # 1. Extract full text
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    full_text = "\n\n".join(pages)

    # 2. Normalize whitespace / paragraphs
    normalized = _normalize_whitespace(full_text)

    # 3. Split into paragraphs
    paragraphs = normalized.split("\n\n")

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue  # skip empty paragraphs

        para_len = len(para)

        # If single paragraph itself is very large, split it hard
        if para_len > max_chars:
            # naive hard split
            start = 0
            while start < para_len:
                end = min(start + max_chars, para_len)
                sub = para[start:end]
                if sub.strip():
                    chunks.append(sub.strip())
                start = end
            # do not add this para to current_chunk further
            continue

        # If adding this paragraph would exceed max_chars → finalize current chunk
        if current_len + para_len + 1 > max_chars and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)
            # Start a new chunk with overlap from previous
            if overlap_chars > 0:
                overlap_text = chunk_text[-overlap_chars:]
                current_chunk = [overlap_text]
                current_len = len(overlap_text)
            else:
                current_chunk = []
                current_len = 0

        # Add paragraph to current chunk
        current_chunk.append(para)
        current_len += para_len + 2  # +2 for the "\n\n" we join with

    # Last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append(chunk_text)

    return chunks
