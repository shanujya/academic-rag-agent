"""Evaluation engine for measuring Naive RAG vs Self-RAG performance."""

import time
from typing import Callable, Any

from google import genai
from google.genai import types

from config import JUDGE_MODEL, JUDGE_BIAS_DISCLAIMER
from src.agent.llm import get_client, _call_with_retry, _parse_json_response


def judge_with_llm(prompt: str) -> dict:
    """Run an evaluation prompt using the designated JUDGE_MODEL with temperature=0.0."""
    client = get_client()

    def _call():
        return client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

    response = _call_with_retry(_call)
    res = _parse_json_response(response.text or "{}")
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        res = res[0]
    return res if isinstance(res, dict) else {}



def evaluate_faithfulness(question: str, context: str, answer: str) -> int:
    """Judge whether every claim in the answer is supported by the context (1 or 0)."""
    if not answer or "No context available" in context and not answer.strip():
        return 0

    prompt = f"""You are an expert evaluator assessing answer faithfulness (groundedness).

Question: {question}

Context provided:
{context[:6000]}

Generated Answer:
{answer}

Evaluate whether ALL factual claims in the generated answer are supported by the context.
Return JSON with keys:
- "faithful": "yes" if every claim is grounded in the context, otherwise "no"
- "reason": short explanation
"""
    result = judge_with_llm(prompt)
    return 1 if result.get("faithful", "no").lower() == "yes" else 0


def evaluate_relevancy(question: str, answer: str) -> int:
    """Judge whether the answer directly addresses the question (1 or 0)."""
    if not answer:
        return 0

    prompt = f"""You are an expert evaluator assessing answer relevancy.

Question: {question}

Generated Answer:
{answer}

Evaluate whether the generated answer directly addresses the user's question.
Return JSON with keys:
- "relevant": "yes" if the answer directly addresses the question, otherwise "no"
- "reason": short explanation
"""
    result = judge_with_llm(prompt)
    return 1 if result.get("relevant", "no").lower() == "yes" else 0


def evaluate_fallback_accuracy(actual_fallback_used: bool, expected_fallback: bool) -> int:
    """Check if actual web fallback usage matches the ground truth label (1 or 0)."""
    return 1 if actual_fallback_used == expected_fallback else 0


def calculate_self_rag_llm_calls(result: dict) -> int:
    """Count actual Self-RAG LLM API calls for one full pipeline run.

    - grade_documents: 1 batched call regardless of chunk count (if any docs were graded).
      Previously this used len(grades) which counted *chunks*, not *API calls*,
      inflating every run's llm_calls by (TOP_K - 1). Fixed to always be 1.
    - Each generation attempt: 1 generate call + 1 grade_generation call.
    - Each retrieve cycle: 1 grade_answer call.
    """
    grades = result.get("grade_documents_result", {}).get("grades", [])
    doc_grade_calls = 1 if grades else 0          # batched: always 1 call, not len(grades)
    gen_retries = result.get("generation_retries", 1)
    ret_cycles = result.get("retrieve_cycles", 1)
    return doc_grade_calls + (gen_retries * 2) + ret_cycles


def evaluate_single_run(item: dict, pipeline_result: dict, is_self_rag: bool = True) -> dict:
    """Evaluate metrics for a single test item execution result.

    Metric independence note
    ------------------------
    ``faithfulness`` is scored by an **external LLM judge** (JUDGE_MODEL) that reads
    the retrieved context and the generated answer independently.

    ``hallucination_intercepted`` is derived from the pipeline's **own internal**
    ``grade_generation_result`` grounding check — the verdict the Self-RAG graph
    itself produced on the final generation attempt. These are kept intentionally
    separate so the two numbers are independently sourced and can diverge, giving
    a meaningful comparison rather than restating the same underlying judge call.
    """
    question = item["question"]
    expected_fallback = item["should_trigger_web_fallback"]

    answer = pipeline_result.get("generation") if is_self_rag else pipeline_result.get("answer", "")
    web_context = pipeline_result.get("web_context", "")
    actual_fallback = bool(web_context and "WebSearch:" in "".join(pipeline_result.get("steps_log", []))) if is_self_rag else False

    # Extract retrieved context
    docs = pipeline_result.get("relevant_documents") if is_self_rag else pipeline_result.get("retrieved_chunks", [])
    if not docs and is_self_rag:
        docs = pipeline_result.get("documents", [])

    context_str = "\n\n".join(d.page_content if hasattr(d, "page_content") else str(d) for d in docs)
    if web_context:
        context_str += f"\n\nWeb Fallback:\n{web_context}"

    # External judge scores (independent of the pipeline's own self-checks)
    faithfulness = evaluate_faithfulness(question, context_str, answer)
    relevancy = evaluate_relevancy(question, answer)

    if is_self_rag:
        fallback_acc = evaluate_fallback_accuracy(actual_fallback, expected_fallback)
        llm_calls = calculate_self_rag_llm_calls(pipeline_result)
    else:
        fallback_acc = None  # Naive RAG has no fallback mechanism — not a comparable metric
        llm_calls = pipeline_result.get("num_llm_calls", 1)

    # Hallucination self-correction — measured from the pipeline's OWN internal
    # grounding check (grade_generation_result), NOT re-derived from the external
    # faithfulness judge.  This keeps the two metrics genuinely independent.
    hallucination_flagged = False
    hallucination_intercepted = False
    if is_self_rag:
        gen_retries = pipeline_result.get("generation_retries", 1)
        steps_log = pipeline_result.get("steps_log", [])
        has_hallucination_log = any("hallucinated" in log for log in steps_log)

        if gen_retries > 1 or has_hallucination_log:
            hallucination_flagged = True
            # Use the pipeline's own final grounding verdict, not the external judge.
            final_grade = pipeline_result.get("grade_generation_result", {})
            grounded_on_final_attempt = str(final_grade.get("grounded", "no")).lower() == "yes"
            if grounded_on_final_attempt:
                hallucination_intercepted = True

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "ground_truth_answer": item["ground_truth_answer"],
        "generated_answer": answer,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "fallback_accuracy": fallback_acc,
        "actual_fallback_used": actual_fallback,
        "expected_fallback": expected_fallback,
        "hallucination_flagged": hallucination_flagged,
        "hallucination_intercepted": hallucination_intercepted,
        "latency_seconds": pipeline_result.get("latency_seconds", 0.0),
        "llm_calls": llm_calls,
        "context_char_count": len(context_str),
        "context_preview": context_str[:300],
        "web_context_used": bool(web_context),
        "web_context_char_count": len(web_context) if web_context else 0,
    }


def sample_dataset(dataset: list[dict], n: int) -> list[dict]:
    """Stratified sample of n items across categories (in_domain, adversarial, complex).

    Distributes n items proportionally across categories, ensuring each category
    is represented. Items within each category are picked from the front of the list
    (deterministic — no random seed required for reproducibility).
    """
    from collections import defaultdict

    if n <= 0 or n >= len(dataset):
        return dataset

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in dataset:
        by_cat[item.get("category", "unknown")].append(item)

    cats = list(by_cat.keys())
    per_cat = max(1, n // len(cats))
    sampled: list[dict] = []
    for cat in cats:
        sampled.extend(by_cat[cat][:per_cat])

    # Top up to exactly n if rounding left us short
    remaining = [item for item in dataset if item not in sampled]
    sampled.extend(remaining[: n - len(sampled)])

    return sampled[:n]


def evaluate_dataset(
    dataset: list[dict],
    pipeline_fn: Callable[[str], dict],
    is_self_rag: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    inter_item_sleep: float = 4.0,
) -> dict:
    """Orchestrate dataset evaluation across all items for a given pipeline.

    Args:
        dataset: List of evaluation items.
        pipeline_fn: Callable that accepts a question string and returns a result dict.
        is_self_rag: Whether the pipeline is Self-RAG (vs. Naive RAG).
        progress_callback: Optional callback(current, total) for progress tracking.
        inter_item_sleep: Seconds to sleep between items to respect free-tier rate limits.
            With batched grading (1 call per question), 4s is sufficient on gemini-flash-lite.
    """
    results = []
    total = len(dataset)

    total_faithfulness = 0
    total_relevancy = 0
    total_fallback_acc = 0
    total_latency = 0.0
    total_llm_calls = 0

    hallucination_flagged_count = 0
    hallucination_intercepted_count = 0

    for i, item in enumerate(dataset, 1):
        start_t = time.time()
        pipeline_res = pipeline_fn(item["question"])
        elapsed = round(time.time() - start_t, 3)
        pipeline_res["latency_seconds"] = elapsed

        eval_res = evaluate_single_run(item, pipeline_res, is_self_rag=is_self_rag)
        results.append(eval_res)

        total_faithfulness += eval_res["faithfulness"]
        total_relevancy += eval_res["relevancy"]
        if eval_res["fallback_accuracy"] is not None:
            total_fallback_acc += eval_res["fallback_accuracy"]
        total_latency += eval_res["latency_seconds"]
        total_llm_calls += eval_res["llm_calls"]

        if eval_res["hallucination_flagged"]:
            hallucination_flagged_count += 1
            if eval_res["hallucination_intercepted"]:
                hallucination_intercepted_count += 1

        if progress_callback:
            progress_callback(i, total)

        # Pace requests to stay within free-tier RPM limits.
        # With batched grading the total LLM calls per item are now:
        #   1 (batch grade) + 1 (generate) + 1 (grade_generation) + 1 (grade_answer) = 4 max
        # Plus 2 judge calls (faithfulness + relevancy) = 6 total max per item.
        if i < total:
            time.sleep(inter_item_sleep)

    avg_latency = round(total_latency / total, 2) if total > 0 else 0.0
    avg_llm_calls = round(total_llm_calls / total, 2) if total > 0 else 0.0

    return {
        "is_self_rag": is_self_rag,
        "total_questions": total,
        "faithfulness_score": total_faithfulness,
        "faithfulness_pct": f"{round((total_faithfulness / total) * 100, 1)}% ({total_faithfulness}/{total})",
        "relevancy_score": total_relevancy,
        "relevancy_pct": f"{round((total_relevancy / total) * 100, 1)}% ({total_relevancy}/{total})",
        "fallback_acc_score": total_fallback_acc if is_self_rag else None,
        "fallback_acc_pct": (
            f"{round((total_fallback_acc / total) * 100, 1)}% ({total_fallback_acc}/{total})"
            if is_self_rag
            else "N/A (no fallback mechanism)"
        ),
        "hallucination_flagged_count": hallucination_flagged_count,
        "hallucination_intercepted_count": hallucination_intercepted_count,
        "hallucination_interception_pct": (
            f"{round((hallucination_intercepted_count / hallucination_flagged_count) * 100, 1)}% ({hallucination_intercepted_count}/{hallucination_flagged_count})"
            if hallucination_flagged_count > 0
            else "N/A (0/0 flagged)"
        ),
        "avg_latency_seconds": avg_latency,
        "avg_llm_calls": avg_llm_calls,
        "disclaimer": JUDGE_BIAS_DISCLAIMER,
        "item_results": results,
    }
