"""Recursive character text splitting with section awareness."""

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    content: str
    metadata: dict


def _split_recursive(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    if separator:
        parts = text.split(separator)
    else:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    chunks: list[str] = []
    current = ""

    for i, part in enumerate(parts):
        piece = part if separator == "" else (part + separator if i < len(parts) - 1 else part)
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(piece) > chunk_size:
                sub_seps = separators[separators.index(separator) + 1 :] if separator in separators else [""]
                chunks.extend(_split_recursive(piece, sub_seps or [""], chunk_size, chunk_overlap))
                current = ""
            else:
                overlap = current[-chunk_overlap:] if chunk_overlap and current else ""
                current = overlap + piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def split_document(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split document text into overlapping chunks, preserving section headers."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _split_recursive(text, separators, chunk_size, chunk_overlap)

    chunks: list[TextChunk] = []
    for idx, content in enumerate(raw_chunks):
        chunks.append(
            TextChunk(
                content=content,
                metadata={"source": source, "chunk_index": idx},
            )
        )
    return chunks
