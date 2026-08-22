## 📊 Evaluation & Benchmark Results

> **Notice**: Disclaimer: Judge uses the same model family as the generator; treat results as relative comparison between naive and Self-RAG, not absolute quality scores.

> Evaluated on **21 questions** (16 hallucinations flagged by Self-RAG, 15 successfully intercepted).

| Metric | Self-RAG | Naive RAG |
| :--- | :--- | :--- |
| **Faithfulness / Groundedness** | `95.2% (20/21)` | `100.0% (21/21)` |
| **Answer Relevancy** | `38.1% (8/21)` | `33.3% (7/21)` |
| **Fallback Trigger Accuracy** | `71.4% (15/21)` | `N/A (no fallback mechanism)` |
| **Hallucination Self-Correction Rate** *(pipeline-reported, from `grade_generation_result`; independent of Faithfulness judge)* | `93.8% (15/16)` | `N/A (No self-correction)` |
| **Avg Latency per Query** | `28.51s` | `5.30s` |
| **Avg LLM Calls per Query** | `6.24` | `1.00` |
