## 📊 Evaluation & Benchmark Results

> **Notice**: Disclaimer: Judge uses the same model family as the generator; treat results as relative comparison between naive and Self-RAG, not absolute quality scores.

> Evaluated on **9 questions** (6 hallucinations flagged by Self-RAG, 6 successfully intercepted).

| Metric | Self-RAG | Naive RAG |
| :--- | :--- | :--- |
| **Faithfulness / Groundedness** | `100.0% (9/9)` | `100.0% (9/9)` |
| **Answer Relevancy** | `55.6% (5/9)` | `44.4% (4/9)` |
| **Fallback Trigger Accuracy** | `88.9% (8/9)` | `66.7% (6/9)` |
| **Hallucination Interception Rate** | `100.0% (6/6)` | `N/A (No reflection)` |
| **Avg Latency per Query** | `38.05s` | `5.13s` |
| **Avg LLM Calls per Query** | `10.00` | `1.00` |
