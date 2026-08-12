"""End-to-end PDF ingestion pipeline."""

from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR
from src.ingestion.chunker import split_document
from src.ingestion.embed_store import embed_and_store, get_vectorstore
from src.ingestion.pdf_parser import extract_pdf_text


def list_paper_pdfs(pdf_dir: Path | None = None) -> list[Path]:
    """Return numbered academic PDFs (01-12) from the paper directory."""
    root = pdf_dir or PDF_DIR
    pdfs = sorted(root.glob("*.pdf"))
    numbered = [p for p in pdfs if p.name[:2].isdigit()]
    return numbered if numbered else pdfs


def ingest_papers(pdf_dir: Path | None = None, reset: bool = False) -> dict:
    """Parse, chunk, embed, and store all papers. Returns ingestion summary."""
    pdfs = list_paper_pdfs(pdf_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir or PDF_DIR}")

    if reset:
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass

    all_chunks = []
    per_file: dict[str, int] = {}

    for pdf_path in pdfs:
        text = extract_pdf_text(pdf_path)
        chunks = split_document(
            text,
            source=pdf_path.name,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        per_file[pdf_path.name] = len(chunks)
        all_chunks.extend(chunks)

    stored = embed_and_store(all_chunks)
    collection = get_vectorstore()

    return {
        "files_processed": len(pdfs),
        "chunks_stored": stored,
        "chunks_per_file": per_file,
        "collection_count": collection.count(),
    }
