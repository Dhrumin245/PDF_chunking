from __future__ import annotations

from pathlib import Path


def extract_text(pdf_path: str | Path) -> str:
    """Extract text from a PDF file."""

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    try:
        from importlib import import_module

        PdfReader = import_module("pypdf").PdfReader
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if normalized:
            pages.append(f"[page {page_number}]\n{normalized}")
    return "\n\n".join(pages)
