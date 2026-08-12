"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = Path("/home/shanujya/llm_paper")
CHROMA_DIR = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "academic_papers"

# Models
EMBED_MODEL = "gemini-embedding-2"
FLASH_MODEL = "gemini-flash-lite-latest"
PRO_MODEL = "gemini-flash-latest"

# Retrieval
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Self-RAG limits
MAX_GENERATION_RETRIES = 2
MAX_RETRIEVE_CYCLES = 2
