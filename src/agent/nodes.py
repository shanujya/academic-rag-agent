"""Self-RAG LangGraph node implementations."""

from langchain_core.documents import Document

from config import MAX_GENERATION_RETRIES, MAX_RETRIEVE_CYCLES, TOP_K
from src.agent.llm import generate_with_pro, grade_with_flash
from src.agent.state import AgentState
from src.ingestion.embed_store import get_vectorstore


def _log(state: AgentState, message: str) -> list[str]:
    return [message]


def retrieve(state: AgentState) -> dict:
    question = state["question"]
    collection = get_vectorstore()

    if collection.count() == 0:
        return {
            "documents": [],
            "steps_log": _log(state, "Retrieve: vector store is empty — run ingestion first."),
        }

    results = collection.query(query_texts=[question], n_results=TOP_K)
    documents: list[Document] = []

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

    return {
        "documents": documents,
        "retrieve_cycles": state.get("retrieve_cycles", 0) + 1,
        "steps_log": _log(
            state,
            f"Retrieve: fetched {len(documents)} chunks from ChromaDB (top-{TOP_K}).",
        ),
    }


def grade_documents(state: AgentState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])
    relevant: list[Document] = []
    grades: list[dict] = []

    for doc in documents:
        prompt = f"""You are grading document relevance for a Self-RAG academic assistant.

Question: {question}

Document chunk (source: {doc.metadata.get('source', 'unknown')}):
{doc.page_content[:2000]}

Return JSON with keys:
- "binary_score": "yes" if the chunk helps answer the question, otherwise "no"
- "reason": one short sentence explaining your decision
"""
        result = grade_with_flash(prompt)
        is_relevant = result.get("binary_score", "no").lower() == "yes"
        grade_entry = {
            "source": doc.metadata.get("source"),
            "relevant": is_relevant,
            "reason": result.get("reason", ""),
            "retrieval_score": doc.metadata.get("retrieval_score"),
        }
        grades.append(grade_entry)
        if is_relevant:
            doc.metadata["grade_reason"] = result.get("reason", "")
            relevant.append(doc)

    msg = (
        f"GradeDocuments: {len(relevant)}/{len(documents)} chunks marked relevant."
        if documents
        else "GradeDocuments: no chunks to grade."
    )

    return {
        "relevant_documents": relevant,
        "grade_documents_result": {"grades": grades, "relevant_count": len(relevant)},
        "steps_log": _log(state, msg),
    }


def web_search(state: AgentState) -> dict:
    question = state["question"]
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(question, max_results=3))
        snippets = "\n\n".join(
            f"- {h.get('title', 'Untitled')}: {h.get('body', '')}" for h in hits
        )
        context = snippets or "No web results found."
        msg = f"WebSearch: fetched {len(hits)} fallback results."
    except Exception as exc:
        context = ""
        msg = f"WebSearch: failed ({exc}). Proceeding without web context."

    return {
        "web_context": context,
        "steps_log": _log(state, msg),
    }


def generate(state: AgentState) -> dict:
    question = state["question"]
    docs = state.get("relevant_documents") or []
    web_context = state.get("web_context", "")

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] ({source})\n{doc.page_content}")

    if web_context:
        context_parts.append(f"[Web fallback]\n{web_context}")

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
    retries = state.get("generation_retries", 0) + 1

    return {
        "generation": answer,
        "generation_retries": retries,
        "steps_log": _log(state, f"Generate: produced answer (attempt {retries})."),
    }


def grade_generation(state: AgentState) -> dict:
    question = state["question"]
    generation = state.get("generation", "")
    docs = state.get("relevant_documents") or []
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Grade whether the generated answer is grounded in the provided facts (Self-RAG hallucination check).

Question: {question}

Ground-truth context from retrieved chunks:
{context[:6000]}

Generated answer:
{generation}

Return JSON with keys:
- "grounded": "yes" if the answer is supported by the context, "no" if hallucinated or unsupported
- "reason": brief explanation
"""

    result = grade_with_flash(prompt)
    grounded = result.get("grounded", "no").lower() == "yes"

    return {
        "grade_generation_result": result,
        "steps_log": _log(
            state,
            f"GradeGeneration: {'grounded' if grounded else 'hallucinated'} — {result.get('reason', '')}",
        ),
    }


def grade_answer(state: AgentState) -> dict:
    question = state["question"]
    generation = state.get("generation", "")

    prompt = f"""Grade whether the answer addresses the user's question (Self-RAG answer relevance check).

Question: {question}

Answer:
{generation}

Return JSON with keys:
- "addresses_question": "yes" if the answer directly addresses the question, otherwise "no"
- "reason": brief explanation
"""

    result = grade_with_flash(prompt)
    addresses = result.get("addresses_question", "no").lower() == "yes"

    return {
        "grade_answer_result": result,
        "steps_log": _log(
            state,
            f"GradeAnswer: {'addresses question' if addresses else 'does not address question'} — {result.get('reason', '')}",
        ),
    }
