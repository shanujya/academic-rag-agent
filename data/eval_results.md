## 📊 Evaluation & Benchmark Results

> **Notice**: Disclaimer: Judge uses the same model family as the generator; treat results as relative comparison between naive and Self-RAG, not absolute quality scores.

> Evaluated on **28 questions** (22 hallucinations flagged by Self-RAG, 20 successfully intercepted).

| Metric | Self-RAG | Naive RAG |
| :--- | :--- | :--- |
| **Faithfulness / Groundedness** | `96.4% (27/28)` | `100.0% (28/28)` |
| **Answer Relevancy** | `28.6% (8/28)` | `17.9% (5/28)` |
| **Fallback Trigger Accuracy** | `71.4% (20/28)` | `N/A (no fallback mechanism)` |
| **Hallucination Self-Correction Rate** *(pipeline-reported, from `grade_generation_result`; independent of Faithfulness judge)* | `90.9% (20/22)` | `N/A (No self-correction)` |
| **Avg Latency per Query** | `28.21s` | `6.90s` |
| **Avg LLM Calls per Query** | `6.39` | `1.00` |
