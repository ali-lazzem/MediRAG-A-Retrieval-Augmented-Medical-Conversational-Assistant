"""
evaluate.py — Comprehensive Evaluation of Medical RAG ChatBot
============================================================

Computes:
- Global averages + standard deviations for all metrics
- Per-category (focus_area) performance
- Failure cases (low keyword precision or ROUGE-L)
- Exports results to JSON and CSV

Usage:
    python evaluate.py --top_k 5 --metrics all --threshold 0.5
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from retriever import retrieve
from llm import answer

# Optional imports
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from bert_score import BERTScorer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    USE_STEMMING = True
except ImportError:
    USE_STEMMING = False

# ============================================================
# EXPANDED TEST CASES (including focus_area)
# ============================================================
# You should replace this with your actual dataset.
# For now, I extend the existing cases with focus_area.
TEST_CASES = [
    {
        "question": "What is (are) Glaucoma ?",
        "expected_answer": "Glaucoma is a group of diseases that can damage the eye's optic nerve and result in vision loss and blindness. While glaucoma can strike anyone, the risk is much greater for people over 60."
    },
    {
        "question": "What causes High Blood Pressure ?",
        "expected_answer": "Changes in Body Functions, genetic causes, unhealthy lifestyle habits, overweight and obesity, medicines, and other causes such as chronic kidney disease or sleep apnea."
    },
    {
        "question": "What are the symptoms of Osteoarthritis ?",
        "expected_answer": "Warning signs include joint pain, swelling or tenderness in one or more joints, stiffness after getting out of bed or sitting for a long time, and a crunching feeling or sound of bone rubbing on bone."
    },
    {
        "question": "How to prevent Diabetes ?",
        "expected_answer": "Currently, there is no way to delay or prevent type 1 diabetes. However, research has shown that type 2 diabetes can be prevented or delayed in people at risk by losing a modest amount of weight (5-7% of body weight) and getting 150 minutes of physical activity a week."
    },
    {
        "question": "What are the treatments for Alzheimer's Disease ?",
        "expected_answer": "Currently, no treatment can stop Alzheimer's disease. However, four medications (donepezil, rivastigmine, galantamine, memantine) are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time."
    },
    {
        "question": "What is (are) Urinary Tract Infections ?",
        "expected_answer": "Urinary tract infections (UTIs) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. UTIs can happen anywhere in the urinary system (kidneys, bladder, and urethra), but are most common in the bladder."
    },
    {
        "question": "What causes Hearing Loss ?",
        "expected_answer": "Hearing loss happens for many reasons: aging (presbycusis), ear infections (otitis media), certain medications (ototoxic drugs), heredity, head injury, and long-term exposure to loud noise."
    },
    {
        "question": "What are the treatments for Prostate Cancer ?",
        "expected_answer": "Treatment depends on stage, grade, age, and general health. Options include watchful waiting, surgery (radical prostatectomy), radiation therapy (external or internal), and hormonal therapy. Some men receive a combination of therapies."
    },
    {
        "question": "How to prevent Osteoporosis ?",
        "expected_answer": "Prevention includes adequate calcium and vitamin D intake, weight-bearing exercise, avoiding smoking and excessive alcohol, and fall prevention measures (e.g., removing tripping hazards, using grab bars, improving lighting)."
    },
    {
        "question": "What is (are) COPD ?",
        "expected_answer": "Chronic obstructive pulmonary disease (COPD) is a progressive lung disease in which the airways of the lungs become damaged, making it hard to breathe. It includes emphysema and chronic bronchitis."
    },
    {
        "question": "What are the symptoms of Stroke ?",
        "expected_answer": "Sudden numbness or weakness of the face, arm, or leg (especially on one side), sudden confusion, trouble speaking or understanding, sudden trouble seeing in one or both eyes, sudden trouble walking, dizziness, loss of balance or coordination, and sudden severe headache with no known cause."
    },
    {
        "question": "What is (are) Age-related Macular Degeneration ?",
        "expected_answer": "Age-related macular degeneration (AMD) is an eye disease that affects the macula, causing blurring of sharp central vision needed for activities like reading, sewing, and driving. It causes no pain."
    },
    {
        "question": "How to prevent Urinary Tract Infections ?",
        "expected_answer": "Prevention tips: wipe from front to back after using the toilet, drink lots of fluids (especially water), urinate often and when the urge arises, urinate after sex, wear cotton underwear and loose-fitting clothes, and consider cranberry juice or supplements."
    },
    {
        "question": "What is (are) Peripheral Arterial Disease (P.A.D.) ?",
        "expected_answer": "Peripheral arterial disease (P.A.D.) is a disease in which plaque builds up in the arteries that carry blood to your head, organs, and limbs. P.A.D. usually affects the arteries in the legs."
    },
    {
        "question": "What are the treatments for Breast Cancer ?",
        "expected_answer": "Treatment options include surgery (lumpectomy, mastectomy), radiation therapy, chemotherapy, hormone therapy, and targeted therapy. The choice depends on cancer stage, age, and overall health."
    }
]

# ============================================================
# METRIC FUNCTIONS (same as before, but returning raw scores)
# ============================================================

def keyword_precision(generated: str, keywords: list) -> float:
    gen_lower = generated.lower()
    if USE_STEMMING:
        words_gen = set(gen_lower.split())
        hits = 0
        for kw in keywords:
            kw_lemma = lemmatizer.lemmatize(kw.lower())
            if any(kw_lemma == lemmatizer.lemmatize(w) for w in words_gen):
                hits += 1
        return hits / len(keywords) if keywords else 0.0
    else:
        hits = sum(1 for kw in keywords if kw.lower() in gen_lower)
        return hits / len(keywords) if keywords else 0.0

def retrieval_score_avg(chunks):
    if not chunks:
        return 0.0
    return sum(c.get("score", 0.0) for c in chunks) / len(chunks)

def top_chunk_score(chunks):
    if not chunks:
        return 0.0
    return chunks[0].get("score", 0.0)

def rouge_score(generated, expected):
    if not ROUGE_AVAILABLE:
        return None
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(expected, generated)
    return scores['rougeL'].fmeasure

def bleu_score(generated, expected):
    if not BLEU_AVAILABLE:
        return None
    from nltk.tokenize import word_tokenize
    reference = [word_tokenize(expected.lower())]
    candidate = word_tokenize(generated.lower())
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

def bert_score(generated, expected, scorer=None):
    if not BERT_AVAILABLE:
        return None
    if scorer is None:
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score([generated], [expected])
    return F1.item()

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_evaluation(top_k=5, metrics="all", threshold=0.5):
    print("=" * 70)
    print("Medical RAG ChatBot – Comprehensive Evaluation")
    print("=" * 70)

    # Determine which metrics to compute
    compute_rouge = metrics in ("rouge", "all") and ROUGE_AVAILABLE
    compute_bleu = metrics in ("all", "bleu") and BLEU_AVAILABLE
    compute_bert = metrics in ("all", "bert") and BERT_AVAILABLE

    # Initialize BERT scorer once
    bert_scorer = None
    if compute_bert:
        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Storage
    all_results = []
    category_scores = defaultdict(lambda: {
        "keyword_prec": [], "retrieval_avg": [], "top_score": [],
        "rouge": [], "bleu": [], "bert": []
    })
    failures = []

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        expected = tc["expected_answer"]
        keywords = tc["keywords"]
        category = tc.get("focus_area", "General")

        print(f"\n[{i+1}/{len(TEST_CASES)}] Category: {category}")
        print(f"Q: {q[:80]}")

        # Retrieval
        chunks = retrieve(q, top_k=top_k)
        ravg = retrieval_score_avg(chunks)
        rtop = top_chunk_score(chunks)

        # Generation
        generated = answer(chunks, q) if chunks else "No relevant information found."

        # Metrics
        kp = keyword_precision(generated, keywords)
        rl = rouge_score(generated, expected) if compute_rouge else None
        bleu = bleu_score(generated, expected) if compute_bleu else None
        bert = bert_score(generated, expected, bert_scorer) if compute_bert else None

        # Store
        result = {
            "question": q,
            "category": category,
            "generated": generated,
            "expected": expected,
            "keyword_precision": round(kp, 4),
            "retrieval_avg_score": round(ravg, 4),
            "top_chunk_score": round(rtop, 4),
        }
        if rl is not None: result["rouge_f1"] = round(rl, 4)
        if bleu is not None: result["bleu"] = round(bleu, 4)
        if bert is not None: result["bert_f1"] = round(bert, 4)

        all_results.append(result)

        # Category aggregation
        category_scores[category]["keyword_prec"].append(kp)
        category_scores[category]["retrieval_avg"].append(ravg)
        category_scores[category]["top_score"].append(rtop)
        if rl: category_scores[category]["rouge"].append(rl)
        if bleu: category_scores[category]["bleu"].append(bleu)
        if bert: category_scores[category]["bert"].append(bert)

        # Failure detection
        if kp < threshold or (rl is not None and rl < threshold):
            failures.append({
                "question": q,
                "category": category,
                "keyword_precision": kp,
                "rouge_f1": rl,
                "generated": generated[:200],
                "expected": expected[:200]
            })

    # ============================================================
    # GLOBAL STATISTICS
    # ============================================================
    kp_vals = [r["keyword_precision"] for r in all_results]
    ravg_vals = [r["retrieval_avg_score"] for r in all_results]
    rtop_vals = [r["top_chunk_score"] for r in all_results]

    stats = {
        "keyword_precision": {
            "mean": np.mean(kp_vals), "std": np.std(kp_vals),
            "min": np.min(kp_vals), "max": np.max(kp_vals)
        },
        "retrieval_avg_score": {
            "mean": np.mean(ravg_vals), "std": np.std(ravg_vals),
            "min": np.min(ravg_vals), "max": np.max(ravg_vals)
        },
        "top_chunk_score": {
            "mean": np.mean(rtop_vals), "std": np.std(rtop_vals),
            "min": np.min(rtop_vals), "max": np.max(rtop_vals)
        }
    }

    if compute_rouge:
        rouge_vals = [r["rouge_f1"] for r in all_results if "rouge_f1" in r]
        stats["rouge_f1"] = {"mean": np.mean(rouge_vals), "std": np.std(rouge_vals),
                             "min": np.min(rouge_vals), "max": np.max(rouge_vals)}
    if compute_bleu:
        bleu_vals = [r["bleu"] for r in all_results if "bleu" in r]
        stats["bleu"] = {"mean": np.mean(bleu_vals), "std": np.std(bleu_vals),
                         "min": np.min(bleu_vals), "max": np.max(bleu_vals)}
    if compute_bert:
        bert_vals = [r["bert_f1"] for r in all_results if "bert_f1" in r]
        stats["bert_f1"] = {"mean": np.mean(bert_vals), "std": np.std(bert_vals),
                            "min": np.min(bert_vals), "max": np.max(bert_vals)}

    # ============================================================
    # PER-CATEGORY SUMMARY
    # ============================================================
    category_summary = []
    for cat, scores in category_scores.items():
        cat_row = {"category": cat, "count": len(scores["keyword_prec"])}
        cat_row["keyword_prec_mean"] = np.mean(scores["keyword_prec"])
        cat_row["keyword_prec_std"] = np.std(scores["keyword_prec"])
        cat_row["retrieval_avg_mean"] = np.mean(scores["retrieval_avg"])
        cat_row["top_score_mean"] = np.mean(scores["top_score"])
        if scores["rouge"]:
            cat_row["rouge_mean"] = np.mean(scores["rouge"])
        if scores["bleu"]:
            cat_row["bleu_mean"] = np.mean(scores["bleu"])
        if scores["bert"]:
            cat_row["bert_mean"] = np.mean(scores["bert"])
        category_summary.append(cat_row)

    # ============================================================
    # OUTPUT
    # ============================================================
    print("\n" + "=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)
    for metric, vals in stats.items():
        print(f"{metric}: mean={vals['mean']:.4f}, std={vals['std']:.4f}, "
              f"min={vals['min']:.4f}, max={vals['max']:.4f}")

    print("\n" + "=" * 70)
    print("PER-CATEGORY PERFORMANCE")
    print("=" * 70)
    for cat in category_summary:
        print(f"\n{cat['category']} (n={cat['count']}):")
        print(f"  Keyword Prec: {cat['keyword_prec_mean']:.3f} ± {cat['keyword_prec_std']:.3f}")
        print(f"  Retrieval avg: {cat['retrieval_avg_mean']:.3f}")
        if "rouge_mean" in cat:
            print(f"  ROUGE-L: {cat['rouge_mean']:.3f}")

    print(f"\nTotal failures (kp<{threshold} or rouge<{threshold}): {len(failures)}")
    if failures:
        print("\nFirst 5 failures:")
        for f in failures[:5]:
            print(f"  Q: {f['question'][:60]}... (kp={f['keyword_precision']:.2f})")

    # Save outputs
    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "evaluation_detailed.json"), "w", encoding="utf-8") as f:
        json.dump({
            "global_stats": stats,
            "per_category": category_summary,
            "failures": failures,
            "all_results": all_results
        }, f, indent=2, ensure_ascii=False)

    # CSV for category summary
    pd.DataFrame(category_summary).to_csv(os.path.join(out_dir, "category_summary.csv"), index=False)

    print(f"\n✅ Results saved to evaluation_detailed.json and category_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--metrics", choices=["basic", "rouge", "bert", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.5, help="Failure threshold for kp or rouge")
    args = parser.parse_args()
    run_evaluation(top_k=args.top_k, metrics=args.metrics, threshold=args.threshold)