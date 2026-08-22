## 📊 Evaluation & Benchmark Results

> **Notice**: Disclaimer: Judge uses the same model family as the generator; treat results as relative comparison between naive and Self-RAG, not absolute quality scores.

> Evaluated on **21 questions** (15 hallucinations flagged by Self-RAG, 15 successfully intercepted).

| Metric | Self-RAG | Naive RAG |
| :--- | :--- | :--- |
| **Faithfulness / Groundedness** | `100.0% (21/21)` | `100.0% (21/21)` |
| **Answer Relevancy** | `42.9% (9/21)` | `33.3% (7/21)` |
| **Fallback Trigger Accuracy** | `66.7% (14/21)` | `N/A (no fallback mechanism)` |
| **Hallucination Self-Correction Rate** *(pipeline-reported, from `grade_generation_result`; independent of Faithfulness judge)* | `100.0% (15/15)` | `N/A (No self-correction)` |
| **Avg Latency per Query** | `19.04s` | `3.99s` |
| **Avg LLM Calls per Query** | `6.14` | `1.00` |
