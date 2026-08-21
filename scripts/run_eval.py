#!/usr/bin/env python3
"""CLI script to run quantitative evaluations comparing Naive RAG vs Self-RAG.

Usage examples
--------------
# Full run — Self-RAG only
python scripts/run_eval.py

# Quick test — 9 questions stratified across categories, both pipelines
python scripts/run_eval.py --sample 9 --compare-naive --save-results

# Full run comparing both pipelines, saves JSON + Markdown report
python scripts/run_eval.py --compare-naive --save-results
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import JUDGE_BIAS_DISCLAIMER
from src.agent.evaluator import evaluate_dataset, sample_dataset
from src.agent.graph import run_agent
from src.agent.naive_rag import run_naive_rag


def print_summary_table(self_rag_eval: dict, naive_eval: dict | None = None):
    """Print ASCII comparison summary table to stdout."""
    print("\n" + "=" * 80)
    print(" 📊 ACADEMIC SELF-RAG EVALUATION SUMMARY")
    print("=" * 80)
    print(f" {JUDGE_BIAS_DISCLAIMER}")
    print("-" * 80)

    header = f"{'Metric':<32} | {'Self-RAG':<22}"
    if naive_eval:
        header += f" | {'Naive RAG':<20}"
    print(header)
    print("-" * 80)

    # Faithfulness
    row_faith = f"{'Faithfulness / Groundedness':<32} | {self_rag_eval['faithfulness_pct']:<22}"
    if naive_eval:
        row_faith += f" | {naive_eval['faithfulness_pct']:<20}"
    print(row_faith)

    # Relevancy
    row_rel = f"{'Answer Relevancy':<32} | {self_rag_eval['relevancy_pct']:<22}"
    if naive_eval:
        row_rel += f" | {naive_eval['relevancy_pct']:<20}"
    print(row_rel)

    # Fallback Accuracy
    row_fall = f"{'Fallback Trigger Accuracy':<32} | {self_rag_eval['fallback_acc_pct']:<22}"
    if naive_eval:
        row_fall += f" | {naive_eval['fallback_acc_pct']:<20}"
    print(row_fall)

    # Hallucination Self-Correction Rate (pipeline-reported)
    row_hall = f"{'Hallucination Self-Correction Rate':<32} | {self_rag_eval['hallucination_interception_pct']:<22}"
    if naive_eval:
        row_hall += f" | {'N/A (no self-correction)':<20}"
    print(row_hall)

    # Latency
    row_lat = f"{'Avg Latency (seconds)':<32} | {self_rag_eval['avg_latency_seconds']:<22.2f}"
    if naive_eval:
        row_lat += f" | {naive_eval['avg_latency_seconds']:<20.2f}"
    print(row_lat)

    # LLM Calls
    row_calls = f"{'Avg LLM Calls / Query':<32} | {self_rag_eval['avg_llm_calls']:<22.2f}"
    if naive_eval:
        row_calls += f" | {naive_eval['avg_llm_calls']:<20.2f}"
    print(row_calls)

    print("=" * 80 + "\n")


def generate_markdown_report(self_rag_eval: dict, naive_eval: dict | None = None) -> str:
    """Generate Markdown summary table string."""
    n_q = self_rag_eval["total_questions"]
    n_flagged = self_rag_eval["hallucination_flagged_count"]
    n_intercepted = self_rag_eval["hallucination_intercepted_count"]

    md = []
    md.append("## 📊 Evaluation & Benchmark Results\n")
    md.append(f"> **Notice**: {JUDGE_BIAS_DISCLAIMER}\n")
    md.append(f"> Evaluated on **{n_q} questions** "
              f"({n_flagged} hallucinations flagged by Self-RAG, "
              f"{n_intercepted} successfully intercepted).\n")

    md.append("| Metric | Self-RAG | Naive RAG |")
    md.append("| :--- | :--- | :--- |")

    faith_naive = naive_eval["faithfulness_pct"] if naive_eval else "N/A"
    rel_naive = naive_eval["relevancy_pct"] if naive_eval else "N/A"
    fall_naive = naive_eval["fallback_acc_pct"] if naive_eval else "N/A"
    lat_naive = f"{naive_eval['avg_latency_seconds']:.2f}s" if naive_eval else "N/A"
    calls_naive = f"{naive_eval['avg_llm_calls']:.2f}" if naive_eval else "N/A"

    md.append(f"| **Faithfulness / Groundedness** | `{self_rag_eval['faithfulness_pct']}` | `{faith_naive}` |")
    md.append(f"| **Answer Relevancy** | `{self_rag_eval['relevancy_pct']}` | `{rel_naive}` |")
    md.append(f"| **Fallback Trigger Accuracy** | `{self_rag_eval['fallback_acc_pct']}` | `{fall_naive}` |")
    md.append(f"| **Hallucination Self-Correction Rate** *(pipeline-reported, from `grade_generation_result`; independent of Faithfulness judge)* | `{self_rag_eval['hallucination_interception_pct']}` | `N/A (No self-correction)` |")
    md.append(f"| **Avg Latency per Query** | `{self_rag_eval['avg_latency_seconds']:.2f}s` | `{lat_naive}` |")
    md.append(f"| **Avg LLM Calls per Query** | `{self_rag_eval['avg_llm_calls']:.2f}` | `{calls_naive}` |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmark for Academic Self-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT / "data" / "eval_dataset.json",
        help="Path to evaluation dataset JSON file (default: data/eval_dataset.json)",
    )
    parser.add_argument(
        "--compare-naive",
        action="store_true",
        help="Run baseline Naive RAG evaluation alongside Self-RAG",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to data/eval_results.json and data/eval_results.md",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Stratified sample of N questions across categories for a quick test run. "
            "0 (default) means use the full dataset."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        metavar="SECONDS",
        help=(
            "Seconds to sleep between evaluation items (default: 4.0). "
            "Increase if you hit rate limits; decrease on paid tiers."
        ),
    )
    args = parser.parse_args()

    if not args.dataset_path.exists():
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Optionally subsample for quick testing
    if args.sample > 0:
        dataset = sample_dataset(dataset, args.sample)
        print(f"\n🎲 Sampled {len(dataset)} questions (stratified across categories) from dataset.")

    total_q = len(dataset)
    print(f"\n🚀 Starting evaluation benchmark on {total_q} questions...")
    print(f"   ⏱  Inter-item sleep: {args.sleep}s  |  Batched chunk grading: ON\n")

    # ── Self-RAG evaluation ──────────────────────────────────────────────────
    print("[1/2] Evaluating Self-RAG Pipeline...")

    def _progress_self(current, total):
        print(f"  ✓ Processed {current}/{total} items (Self-RAG)")

    self_rag_eval = evaluate_dataset(
        dataset=dataset,
        pipeline_fn=run_agent,
        is_self_rag=True,
        progress_callback=_progress_self,
        inter_item_sleep=args.sleep,
    )

    # ── Naive RAG evaluation (optional) ─────────────────────────────────────
    naive_eval = None
    if args.compare_naive:
        print("\n[2/2] Evaluating Baseline Naive RAG Pipeline...")

        def _progress_naive(current, total):
            print(f"  ✓ Processed {current}/{total} items (Naive RAG)")

        naive_eval = evaluate_dataset(
            dataset=dataset,
            pipeline_fn=run_naive_rag,
            is_self_rag=False,
            progress_callback=_progress_naive,
            inter_item_sleep=args.sleep,
        )
    else:
        print("\n[2/2] Skipped Naive RAG (use --compare-naive to enable).")

    print_summary_table(self_rag_eval, naive_eval)

    # ── Save results ─────────────────────────────────────────────────────────
    if args.save_results:
        out_json_path = ROOT / "data" / "eval_results.json"
        out_md_path = ROOT / "data" / "eval_results.md"

        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": str(args.dataset_path),
            "sample_n": args.sample if args.sample > 0 else total_q,
            "inter_item_sleep": args.sleep,
            "self_rag_summary": {k: v for k, v in self_rag_eval.items() if k != "item_results"},
            "naive_rag_summary": (
                {k: v for k, v in naive_eval.items() if k != "item_results"}
                if naive_eval else None
            ),
            "self_rag_item_results": self_rag_eval["item_results"],
            "naive_rag_item_results": naive_eval["item_results"] if naive_eval else [],
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        md_report = generate_markdown_report(self_rag_eval, naive_eval)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_report + "\n")

        print(f"✅ Saved results to:")
        print(f"  - {out_json_path}")
        print(f"  - {out_md_path}")


if __name__ == "__main__":
    main()
