"""Streamlit dashboard for the Self-RAG academic agent."""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from config import CHROMA_DIR, PDF_DIR
from src.agent.graph import run_agent
from src.ingestion.embed_store import get_vectorstore
from src.ingestion.ingest import ingest_papers, list_paper_pdfs

st.set_page_config(
    page_title="Academic Self-RAG Agent",
    page_icon="📚",
    layout="wide",
)

st.title("Multi-Paper Academic RAG Agent (Self-RAG)")
st.caption("LangGraph + ChromaDB + Gemini · grounded answers over your paper library")


def _check_api_key() -> bool:
    import os

    if os.environ.get("GEMINI_API_KEY"):
        return True
    st.error(
        "GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and add your key, "
        "or run: `export GEMINI_API_KEY=your_key`"
    )
    return False


with st.sidebar:
    st.header("Control Panel")

    pdfs = list_paper_pdfs()
    st.metric("PDFs in library", len(pdfs))
    st.caption(f"Source: `{PDF_DIR}`")

    try:
        count = get_vectorstore().count()
    except Exception:
        count = 0
    st.metric("Indexed chunks", count)

    if st.button("Ingest / Re-index Papers", use_container_width=True):
        if not _check_api_key():
            st.stop()
        with st.spinner("Parsing PDFs, chunking, and embedding..."):
            reset = count > 0
            summary = ingest_papers(reset=reset)
        st.success(
            f"Ingested {summary['files_processed']} files → {summary['chunks_stored']} chunks"
        )
        st.rerun()

    st.divider()
    st.subheader("Execution Log")
    log_container = st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your papers (e.g. How does Self-RAG reduce hallucinations?)")

if question:
    if not _check_api_key():
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running Self-RAG graph..."):
            try:
                result = run_agent(question)
                st.session_state.last_run = result
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

        answer = result.get("generation", "")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.last_run:
    result = st.session_state.last_run

    with log_container.container():
        for step in result.get("steps_log", []):
            st.text(step)

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Retrieved Chunks")
        docs = result.get("documents", [])
        if docs:
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("retrieval_score")
                label = f"Chunk {i}: {source}"
                if score is not None:
                    label += f" (score: {score})"
                with st.expander(label):
                    st.text(doc.page_content[:1500])
        else:
            st.info("No chunks retrieved.")

    with col_right:
        st.subheader("Self-Reflection Metrics")

        grade_doc = result.get("grade_documents_result", {})
        grades = grade_doc.get("grades", [])
        if grades:
            df = pd.DataFrame(grades)
            df["relevant"] = df["relevant"].map({True: "yes", False: "no"})
            st.dataframe(df, use_container_width=True, hide_index=True)

            chart_df = df.groupby("relevant").size().reset_index(name="count")
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(x="relevant:N", y="count:Q", color="relevant:N")
                .properties(title="Document Relevance Grades", height=180)
            )
            st.altair_chart(chart, use_container_width=True)

        gen_grade = result.get("grade_generation_result", {})
        ans_grade = result.get("grade_answer_result", {})

        m1, m2, m3 = st.columns(3)
        m1.metric("Relevant chunks", grade_doc.get("relevant_count", 0))
        m2.metric(
            "Grounded",
            gen_grade.get("grounded", "—"),
            help=gen_grade.get("reason", ""),
        )
        m3.metric(
            "Addresses question",
            ans_grade.get("addresses_question", "—"),
            help=ans_grade.get("reason", ""),
        )

        if result.get("web_context"):
            with st.expander("Web fallback context"):
                st.text(result["web_context"])
