"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = Path("/home/shanujya/llm_paper")
CHROMA_DIR = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "academic_papers"

# Models
EMBED_MODEL = "gemini-embedding-2"
FLASH_MODEL = "gemini-3.5-flash-lite"
PRO_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = "gemini-3.5-flash-lite"





JUDGE_BIAS_DISCLAIMER = (
    "Disclaimer: Judge uses the same model family as the generator; "
    "treat results as relative comparison between naive and Self-RAG, "
    "not absolute quality scores."
)

# Retrieval
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Self-RAG limits
MAX_GENERATION_RETRIES = 2
MAX_RETRIEVE_CYCLES = 2

