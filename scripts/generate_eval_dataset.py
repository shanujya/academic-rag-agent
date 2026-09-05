#!/usr/bin/env python3
"""Automated Synthetic Dataset Generator for Academic Self-RAG.

Scans PDF_DIR for new academic papers not yet present in data/eval_dataset.json,
uses Gemini to generate structured in_domain and complex evaluation questions,
and appends them to the dataset.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from config import PDF_DIR
from src.agent.llm import grade_with_flash


def extract_paper_summary(pdf_path: Path, max_pages: int = 4) -> str:
    """Extract text from the first N pages of a PDF paper."""
    doc = pymupdf.open(str(pdf_path))
    text_chunks = []
    num_pages = min(len(doc), max_pages)
    for i in range(num_pages):
        text = doc[i].get_text("text")
        if text.strip():
            text_chunks.append(f"--- Page {i+1} ---\n{text}")
    doc.close()
    return "\n\n".join(text_chunks)


def get_indexed_papers(dataset: list[dict]) -> set[str]:
    """Extract all PDF filenames already tagged in the dataset."""
    indexed = set()
    for item in dataset:
        for fname in item.get("relevant_source_files", []):
            if fname:
                indexed.add(fname)
    return indexed


def generate_qa_for_paper(pdf_name: str, paper_text: str) -> list[dict]:
    """Call Gemini to generate in_domain and complex Q&A pairs for a paper."""
    prompt = f"""You are an expert AI researcher building a benchmark evaluation dataset for a RAG system.
Given the text extract of the academic paper "{pdf_name}", generate 2 evaluation question-answer pairs:

1. ONE "in_domain" question: Target a specific mathematical formula, key mechanism, or architectural contribution unique to this paper.
2. ONE "complex" question: Compare or synthesize a key technique from this paper with other foundational AI models (such as Transformers, BERT, or Attention).

Requirements:
- Questions must be precise, challenging, and technical.
- Ground truth answers must be detailed and accurate.
- Output MUST be a JSON dictionary containing a "items" list of 2 objects:

{{
  "items": [
    {{
      "category": "in_domain",
      "question": "<Precise question>",
      "ground_truth_answer": "<Detailed accurate answer>",
      "should_trigger_web_fallback": false,
      "notes": "<Brief note on what this evaluates>"
    }},
    {{
      "category": "complex",
      "question": "<Synthesizing comparative question>",
      "ground_truth_answer": "<Detailed accurate answer>",
      "should_trigger_web_fallback": false,
      "notes": "<Brief note on comparison>"
    }}
  ]
}}

Paper Extract:
{paper_text[:8000]}
"""

    res = grade_with_flash(prompt)
    if isinstance(res, dict) and "items" in res:
        qa_pairs = res["items"]
    elif isinstance(res, list):
        qa_pairs = res
    else:
        qa_pairs = []

    for item in qa_pairs:
        item["relevant_source_files"] = [pdf_name]
        
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Auto-generate eval Q&A items for un-indexed PDFs")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PDF_DIR,
        help=f"Directory containing PDF papers (default: {PDF_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_dataset.json",
        help="Path to eval_dataset.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate questions and print them without saving to file",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file not found at {args.dataset}")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    indexed_papers = get_indexed_papers(dataset)
    print(f"Loaded dataset with {len(dataset)} items. Found {len(indexed_papers)} indexed PDF filenames.")

    # Find unindexed PDFs
    all_pdfs = [p for p in args.pdf_dir.glob("*.pdf") if p.is_file()]
    unindexed_pdfs = [p for p in all_pdfs if p.name not in indexed_papers]

    if not unindexed_pdfs:
        print("✅ All PDFs in your paper directory are already indexed in the evaluation dataset!")
        return

    print(f"Found {len(unindexed_pdfs)} un-indexed PDF(s): {[p.name for p in unindexed_pdfs]}")

    next_id_num = len(dataset) + 1
    new_items = []

    for pdf_path in unindexed_pdfs:
        print(f"\n📄 Processing: {pdf_path.name}...")
        try:
            paper_text = extract_paper_summary(pdf_path)
            qa_pairs = generate_qa_for_paper(pdf_path.name, paper_text)
            
            for item in qa_pairs:
                item_id = f"q{next_id_num:03d}"
                next_id_num += 1
                
                full_item = {
                    "id": item_id,
                    "category": item.get("category", "in_domain"),
                    "question": item["question"],
                    "ground_truth_answer": item["ground_truth_answer"],
                    "should_trigger_web_fallback": item.get("should_trigger_web_fallback", False),
                    "relevant_source_files": item.get("relevant_source_files", [pdf_path.name]),
                    "notes": item.get("notes", f"Auto-generated for {pdf_path.name}"),
                }
                new_items.append(full_item)
                print(f"  + Generated {full_item['id']} ({full_item['category']}): {full_item['question'][:70]}...")

        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")

    if not new_items:
        print("No new questions generated.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Generated {len(new_items)} new items. Showing preview:")
        print(json.dumps(new_items, indent=2))
    else:
        dataset.extend(new_items)
        with open(args.dataset, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"\n🎉 Successfully appended {len(new_items)} new items! Updated total dataset size: N = {len(dataset)}.")


if __name__ == "__main__":
    main()
