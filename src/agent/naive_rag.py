"""Minimal baseline Naive RAG pipeline (no reflection, no grading, no retries)."""

import time
from langchain_core.documents import Document

from config import TOP_K
from src.agent.llm import generate_with_pro
from src.ingestion.embed_store import get_vectorstore


def run_naive_rag(question: str) -> dict:
    """Execute a single-pass Naive RAG pipeline for baseline comparison."""
    start_time = time.time()
    num_llm_calls = 0

    collection = get_vectorstore()

    if collection.count() == 0:
        documents: list[Document] = []
    else:
        results = collection.query(query_texts=[question], n_results=TOP_K)
        documents = []
        for idx, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx]
            distance = results["distances"][0][idx] if results.get("distances") else None
            documents.append(
                Document(
                    page_content=doc_text,
                    metadata={
                        **meta,
                        "retrieval_score": round(1 - distance, 4) if distance is not None else None,
                    },
                )
            )

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] ({source})\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

    prompt = f"""You are an academic research assistant answering questions about ML/NLP papers.

Use ONLY the provided context. If the context is insufficient, say what is missing.
Cite sources inline using the paper filename in parentheses, e.g. (10_self-rag_...pdf).

Question:
{question}

Context:
{context}

Write a clear, accurate, well-structured answer with citations where possible.
"""

    answer = generate_with_pro(prompt)
    num_llm_calls += 1
    latency = round(time.time() - start_time, 3)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": documents,
        "latency_seconds": latency,
        "num_llm_calls": num_llm_calls,
        "web_context": "",
    }
