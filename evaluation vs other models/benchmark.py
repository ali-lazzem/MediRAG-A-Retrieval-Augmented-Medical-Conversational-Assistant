#!/usr/bin/env python
"""Benchmark runner using a curated, diverse set of medical questions (RAG only)."""

import argparse
import json
import time
import os
from typing import Dict, Any

from evaluator import compute_semantic_similarity
from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25Retriever
from models.rag_model import RAGModel
from config import DEFAULT_TOP_K, EXISTING_FAISS_INDEX
from data_loader import get_curated_test_set, load_dataset, build_documents


def run_benchmark(csv_path: str, top_k: int) -> Dict[str, Any]:
    """Run benchmark using the curated test set (only RAG models)."""
    # Load curated test set directly
    test_dataset = get_curated_test_set(csv_path)
    print(f"Running benchmark on {len(test_dataset)} questions.\n")

    # Build full documents for BM25 (using the whole CSV, not just test set)
    full_dataset = load_dataset(csv_path)
    all_docs = build_documents(full_dataset)
    all_metadata = [
        {
            "doc_id": i,
            "text": all_docs[i],
            "question": item["question"],
            "answer": item["answer"],
        }
        for i, item in enumerate(full_dataset)
    ]

    # ---- Initialise RAG models ----
    # RAG + FAISS (existing pre-built index)
    faiss_ret = FaissRetriever(load_existing_index=True)
    rag_faiss = RAGModel(faiss_ret)

    # RAG + BM25 (build from documents)
    bm25_ret = BM25Retriever()
    bm25_ret.build_index(all_docs, all_metadata)
    rag_bm25 = RAGModel(bm25_ret)

    models = [rag_faiss, rag_bm25]

    results = {model.name: {"scores": [], "latencies": []} for model in models}

    for idx, test_item in enumerate(test_dataset):
        question = test_item["question"]
        gold_answer = test_item["answer"]
        print(f"Processing [{idx+1}/{len(test_dataset)}]: {question[:80]}...")

        for model in models:
            start = time.perf_counter()
            generated = model.generate(question, top_k=top_k)
            end = time.perf_counter()
            latency = end - start
            similarity = compute_semantic_similarity(generated, gold_answer)

            results[model.name]["scores"].append(similarity)
            results[model.name]["latencies"].append(latency)

    # Aggregate statistics
    summary = []
    for model_name, data in results.items():
        scores = data["scores"]
        latencies = data["latencies"]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        summary.append({
            "model": model_name,
            "mean_semantic_similarity": round(mean_score, 4),
            "avg_latency_seconds": round(mean_latency, 3),
            "num_questions": len(scores)
        })

    # Include the test questions and gold answers for reference
    return {
        "summary": summary,
        "detailed": results,
        "test_questions": [{"question": item["question"], "gold_answer": item["answer"]}
                           for item in test_dataset]
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RAG models on a curated medical QA set.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to medquad.csv (must contain 'question' and 'answer' columns).")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of chunks to retrieve.")
    args = parser.parse_args()

    if not os.path.exists(EXISTING_FAISS_INDEX):
        print(f"ERROR: Existing FAISS index not found at {EXISTING_FAISS_INDEX}")
        print("Please update config.EXISTING_FAISS_INDEX to point to your csv_faiss.index file.")
        return

    results = run_benchmark(args.csv, args.top_k)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (RAG only – FAISS vs BM25)")
    print("=" * 60)
    print(f"{'Model':<40} {'Similarity':<12} {'Latency (s)':<12}")
    print("-" * 60)
    for entry in results["summary"]:
        print(f"{entry['model']:<40} {entry['mean_semantic_similarity']:<12} {entry['avg_latency_seconds']:<12}")

    out_file = "benchmark_results_rag_only.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_file}")


if __name__ == "__main__":
    main()