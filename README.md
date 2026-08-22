# 📚 Academic Self-RAG Agent

A multi-paper academic research assistant powered by **Self-RAG** (Self-Reflective Retrieval-Augmented Generation). Ask natural language questions over your personal PDF library and get grounded, cited answers — with automatic hallucination detection and web fallback.

Built with **LangGraph**, **ChromaDB**, and **Google Gemini**.

---

## ✨ Features

- **Self-RAG pipeline** — the agent reflects on its own outputs at every step:
  - 🔍 Retrieves relevant chunks from your PDF library
  - 🧑‍⚖️ Grades each chunk for relevance before using it
  - 🌐 Falls back to DuckDuckGo web search if no relevant chunks are found
  - ✍️ Generates a grounded, cited answer
  - 🔬 Checks whether the answer is hallucinated or supported by context
  - ✅ Checks whether the answer actually addresses the question
  - 🔁 Retries or re-retrieves if grounding/answer checks fail
- **Persistent vector store** using ChromaDB with Gemini embeddings
- **Streamlit dashboard** with chat interface, chunk viewer, and self-reflection metrics
- **Automatic rate-limit retry** with exponential backoff for Gemini API 429 errors

---

## 🏗️ Architecture

```
User Question
     │
     ▼
 [retrieve]  ──→  ChromaDB (cosine similarity, top-K chunks)
     │
     ▼
 [grade_documents]  ──→  LLM grades each chunk: relevant / not relevant
     │
     ├── (relevant chunks found) ──────────────────────┐
     │                                                  │
     └── (no relevant chunks) ──→ [web_search]  ───────┘
                                                        │
                                                        ▼
                                                   [generate]  ──→  LLM writes cited answer
                                                        │
                                                        ▼
                                              [grade_generation]  ──→  Hallucination check
                                                        │
                                            ┌───────────┴───────────┐
                                     (grounded)               (hallucinated, retry)
                                            │                        │
                                            ▼                   [generate]
                                      [grade_answer]  ──→  Does it address the question?
                                            │
                                ┌───────────┴───────────┐
                           (yes / max cycles)         (no, re-retrieve)
                                │                        │
                               END                  [retrieve]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM & Embeddings | [Google Gemini API](https://ai.google.dev/) (`google-genai`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persistent, cosine similarity) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Web fallback | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) |
| UI | [Streamlit](https://streamlit.io/) |
| Visualizations | [Altair](https://altair-viz.github.io/) |

---

## 📁 Project Structure

```
academic-rag-agent/
├── app.py                  # Streamlit dashboard entry point
├── config.py               # Centralized config (models, paths, hyperparameters)
├── requirements.txt
├── .env.example            # Template for environment variables
├── data/
│   ├── eval_dataset.json   # 21-question evaluation dataset (in_domain / adversarial / complex)
│   ├── eval_results.json   # Last benchmark run — full per-item results
│   ├── eval_results.md     # Last benchmark run — Markdown summary table
│   └── chromadb/           # Persistent ChromaDB vector store (git-ignored)
├── scripts/
│   ├── ingest.py           # Standalone CLI ingestion script
│   └── run_eval.py         # CLI benchmark runner (Self-RAG vs Naive RAG)
└── src/
    ├── agent/
    │   ├── graph.py        # LangGraph workflow: nodes, edges, routing logic
    │   ├── nodes.py        # Node implementations (retrieve, grade, generate, etc.)
    │   ├── llm.py          # Gemini client helpers with retry-on-429 logic
    │   ├── evaluator.py    # LLM-as-a-judge evaluation engine
    │   ├── naive_rag.py    # Baseline Naive RAG (retrieve → generate, no reflection)
    │   └── state.py        # AgentState TypedDict
    └── ingestion/
        ├── pdf_parser.py   # PDF → raw text via PyMuPDF
        ├── chunker.py      # Text chunking with overlap
        ├── embed_store.py  # Gemini embedding function + ChromaDB upsert
        └── ingest.py       # End-to-end ingestion pipeline
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
- PDF papers stored in a local directory (default: `/home/<user>/llm_paper`)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/academic-rag-agent.git
cd academic-rag-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
```

> ⚠️ **Never commit `.env` to git.** It is already listed in `.gitignore`.

### 5. Update the PDF directory path

Edit [`config.py`](config.py) and set `PDF_DIR` to your papers folder:

```python
PDF_DIR = Path("/path/to/your/pdf/papers")
```

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

### Step 1 — Ingest your papers

In the **sidebar**, click **"Ingest / Re-index Papers"**. This will:
1. Parse all PDFs in `PDF_DIR`
2. Chunk the text (1000 chars, 200 overlap)
3. Generate Gemini embeddings and store them in ChromaDB

### Step 2 — Ask questions

Type your question in the chat input at the bottom, e.g.:

> *"How does Self-RAG reduce hallucinations compared to standard RAG?"*

### Step 3 — Inspect the results

After each response, the dashboard shows:
- **Retrieved Chunks** — the top-K chunks fetched from ChromaDB with their similarity scores
- **Self-Reflection Metrics** — document relevance grades, groundedness check, and answer relevance check

---

## ⚙️ Configuration

All key parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PDF_DIR` | `/home/<user>/llm_paper` | Directory containing your PDF papers |
| `EMBED_MODEL` | `gemini-embedding-2` | Gemini embedding model |
| `FLASH_MODEL` | `gemini-flash-lite-latest` | Fast model used for grading/reflection |
| `PRO_MODEL` | `gemini-flash-latest` | Smarter model used for answer generation |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `MAX_GENERATION_RETRIES` | `2` | Max re-generation attempts on hallucination |
| `MAX_RETRIEVE_CYCLES` | `2` | Max re-retrieval cycles on poor answer |

---

## 🔑 API Rate Limits

This project uses the **Gemini API free tier**. Rate limits apply per model per minute. The agent includes automatic retry-with-backoff logic — if you hit a `429 RESOURCE_EXHAUSTED` error, the agent will wait the suggested retry delay and try again (up to 4 times).

To check your project's current live rate limits: [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit)

To remove rate limit constraints, link a billing account in [Google AI Studio](https://aistudio.google.com/) to upgrade to Tier 1.

---

## 📦 Dependencies

```
google-genai>=1.0.0
langgraph>=0.2.0
langchain-core>=0.3.0
chromadb>=0.5.0
pymupdf>=1.24.0
streamlit>=1.38.0
pandas>=2.0.0
altair>=5.0.0
duckduckgo-search>=6.0.0
python-dotenv>=1.0.0
```

---

## 📊 Evaluation & Benchmarking

The system includes an automated evaluation engine to quantitatively compare **Self-RAG** against a baseline **Naive RAG** implementation (`query → retrieve(top_k) → generate`, no reflection loops).

### Evaluation Methodology

- **Test Dataset**: 21 curated questions across 3 categories:
  - `in_domain` (7): Directly answered by PDF library papers.
  - `adversarial` (7): Topics outside the paper library — tests web-search fallback trigger accuracy.
  - `complex` (7): Multi-paper synthesis and architectural comparisons.
- **LLM-as-a-Judge**: Faithfulness and Answer Relevancy scored by `JUDGE_MODEL` at `temperature=0.0`.
- **Self-Correction metric**: Independently sourced from the pipeline's own `grade_generation_result` verdict — **not** re-derived from the Faithfulness judge, so the two numbers are genuinely independent.
- **Batched chunk grading**: All `TOP_K` retrieved chunks are graded in a **single** LLM call, reducing per-query API calls from `TOP_K + 3` to `4` in the typical case.

### Results (N = 21, run 2026-08-22)

> ⚠️ **Disclaimer**: Judge uses the same model family as the generator; treat results as a relative comparison, not absolute quality scores.

| Metric | Self-RAG | Naive RAG |
| :--- | :---: | :---: |
| **Faithfulness / Groundedness** | `100.0% (21/21)` | `100.0% (21/21)` |
| **Answer Relevancy** | `42.9% (9/21)` | `33.3% (7/21)` |
| **Fallback Trigger Accuracy** | `66.7% (14/21)` | `N/A (no fallback mechanism)` |
| **Hallucination Self-Correction Rate** *(pipeline-reported)* | `100.0% (15/15)` | N/A (no self-correction) |
| **Avg Latency per Query** | `19.04s` | `3.99s` |
| **Avg LLM Calls per Query** | `6.14` | `1.00` |

**Honest interpretation of results:**

- **Faithfulness is near-ceiling for both pipelines** on this corpus (`100%`). This is a legitimate finding, not a tuning artifact: the ChromaDB index retrieves highly relevant chunks for most in-domain and complex questions, giving the generator strong grounding signal regardless of reflection loops. Self-RAG's advantage on this corpus shows up primarily in **answer relevancy (+9.6 pp)** and **self-correction** (15 of 21 runs triggered at least one hallucination flag; 15 of 15 were resolved on retry), not in faithfulness separation.
- **Fallback Trigger Accuracy is Self-RAG only.** Naive RAG has no relevance-grading or web-search-fallback mechanism, so there is no equivalent decision to score — it always answers from local retrieval alone, regardless of whether the question is in-domain.
- **Latency and call count** reflect the cost of reflection: Self-RAG averages `6.14` LLM calls/query vs `1.00` for Naive RAG — each reflection loop (grade_documents + generate + grade_generation + grade_answer) adds roughly `4` calls.

### Running the Evaluation CLI

```bash
# Full benchmark — both pipelines, saves JSON + Markdown results
python scripts/run_eval.py --compare-naive --save-results

# Quick stratified smoke-test (3 questions per category, ~5 min)
python scripts/run_eval.py --sample 9 --compare-naive --save-results

# Tune pacing for paid tiers (lower sleep = faster)
python scripts/run_eval.py --compare-naive --save-results --sleep 1
```

Results are saved to [`data/eval_results.json`](data/eval_results.json) and [`data/eval_results.md`](data/eval_results.md).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---


