"""Gemini embeddings + persistent ChromaDB storage."""

import os
from typing import Sequence

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
from src.ingestion.chunker import TextChunk


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str | None = None):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Export it or add it to a .env file."
            )
        self.client = genai.Client(api_key=resolved_key)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            result = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
            embeddings.append(result.embeddings[0].values)
        return embeddings


def get_vectorstore():
    """Return a persistent ChromaDB collection with Gemini embeddings."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(chunks: Sequence[TextChunk], batch_size: int = 32) -> int:
    """Embed chunks and upsert into ChromaDB. Returns number of chunks stored."""
    if not chunks:
        return 0

    collection = get_vectorstore()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        source = chunk.metadata["source"]
        idx = chunk.metadata["chunk_index"]
        chunk_id = f"{source}::{idx}"
        ids.append(chunk_id)
        documents.append(chunk.content)
        metadatas.append({**chunk.metadata, "char_count": len(chunk.content)})

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(chunks)
