# rag/loaders/pdf_loader.py
import pathlib
from typing import List
from pypdf import PdfReader  # pip install pypdf

def load_pdf_as_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n\n".join(chunks)
