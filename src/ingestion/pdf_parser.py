"""Layout-aware PDF text extraction using PyMuPDF."""

from pathlib import Path

import pymupdf


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract reading-order text from a PDF, respecting multi-column layout."""
    doc = pymupdf.open(pdf_path)
    pages: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")
        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        page_text = "\n\n".join(b[4].strip() for b in blocks if b[4].strip())
        if page_text:
            pages.append(page_text)

    doc.close()
    return "\n\n".join(pages)
