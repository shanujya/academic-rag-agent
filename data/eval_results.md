## 📊 Evaluation & Benchmark Results

> **Notice**: Disclaimer: Judge uses the same model family as the generator; treat results as relative comparison between naive and Self-RAG, not absolute quality scores.

> Evaluated on **21 questions** (15 hallucinations flagged by Self-RAG, 14 successfully intercepted).

| Metric | Self-RAG | Naive RAG |
| :--- | :--- | :--- |
| **Faithfulness / Groundedness** | `95.2% (20/21)` | `100.0% (21/21)` |
| **Answer Relevancy** | `42.9% (9/21)` | `33.3% (7/21)` |
| **Fallback Trigger Accuracy** | `66.7% (14/21)` | `66.7% (14/21)` |
| **Hallucination Self-Correction Rate** *(pipeline-reported, from `grade_generation_result`; independent of Faithfulness judge)* | `93.3% (14/15)` | `N/A (No self-correction)` |
| **Avg Latency per Query** | `27.65s` | `7.84s` |
| **Avg LLM Calls per Query** | `6.14` | `1.00` |
