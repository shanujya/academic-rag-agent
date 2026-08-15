#!/usr/bin/env python3
"""CLI script to ingest academic PDFs into ChromaDB."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.ingest import ingest_papers


def main():
    parser = argparse.ArgumentParser(description="Ingest academic PDFs into ChromaDB")
    parser.add_argument(
        "--pdf-dir",
        type=Path, 
        default=None,
        help="Directory containing PDFs (default: /home/shanujya/llm_paper)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before re-ingesting",
    )
    args = parser.parse_args()

    summary = ingest_papers(pdf_dir=args.pdf_dir, reset=args.reset)
    print("Ingestion complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
