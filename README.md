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
│   └── chromadb/           # Persistent ChromaDB vector store (git-ignored)
├── scripts/
│   └── ingest.py           # Standalone CLI ingestion script
└── src/
    ├── agent/
    │   ├── graph.py        # LangGraph workflow: nodes, edges, routing logic
    │   ├── nodes.py        # Node implementations (retrieve, grade, generate, etc.)
    │   ├── llm.py          # Gemini client helpers with retry-on-429 logic
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

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

