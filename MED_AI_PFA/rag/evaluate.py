"""
evaluate_simple.py — Simplified Semantic Evaluation for Medical RAG ChatBot
==========================================================================
Computes:
- Semantic similarity between generated and expected answers (cosine)
- Retrieval average and top chunk scores
- Per-category performance
- Exports results to evaluation.json and evaluation.csv in MED_AI_PFA/Data/evaluation

Usage:
    python evaluate_simple.py --top_k 5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from retriever import retrieve
from llm import answer

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Determine project root (assuming this script is in rag/ or similar)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to MED_AI_PFA (adjust if needed)
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)   # assuming script is in rag/ under project root
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "Data", "evaluation")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ============================================================
# TEST CASES (unchanged)
# ============================================================
TEST_CASES = [
    {
        "question": "What is (are) Glaucoma ?",
        "expected_answer": "Glaucoma is a group of diseases that can damage the eye's optic nerve and result in vision loss and blindness. While glaucoma can strike anyone, the risk is much greater for people over 60.",
        "focus_area": "Glaucoma"
    },
    {
        "question": "What causes High Blood Pressure ?",
        "expected_answer": "Changes in Body Functions, genetic causes, unhealthy lifestyle habits, overweight and obesity, medicines, and other causes such as chronic kidney disease or sleep apnea.",
        "focus_area": "High Blood Pressure"
    },
    {
        "question": "What are the symptoms of Osteoarthritis ?",
        "expected_answer": "Warning signs include joint pain, swelling or tenderness in one or more joints, stiffness after getting out of bed or sitting for a long time, and a crunching feeling or sound of bone rubbing on bone.",
        "focus_area": "Osteoarthritis"
    },
    {
        "question": "How to prevent Diabetes ?",
        "expected_answer": "Currently, there is no way to delay or prevent type 1 diabetes. However, research has shown that type 2 diabetes can be prevented or delayed in people at risk by losing a modest amount of weight (5-7% of body weight) and getting 150 minutes of physical activity a week.",
        "focus_area": "Diabetes"
    },
    {
        "question": "What are the treatments for Alzheimer's Disease ?",
        "expected_answer": "Currently, no treatment can stop Alzheimer's disease. However, four medications (donepezil, rivastigmine, galantamine, memantine) are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time.",
        "focus_area": "Alzheimer's Disease"
    },
    {
        "question": "What is (are) Urinary Tract Infections ?",
        "expected_answer": "Urinary tract infections (UTIs) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. UTIs can happen anywhere in the urinary system (kidneys, bladder, and urethra), but are most common in the bladder.",
        "focus_area": "Urinary Tract Infections"
    },
    {
        "question": "What causes Hearing Loss ?",
        "expected_answer": "Hearing loss happens for many reasons: aging (presbycusis), ear infections (otitis media), certain medications (ototoxic drugs), heredity, head injury, and long-term exposure to loud noise.",
        "focus_area": "Hearing Loss"
    },
    {
        "question": "What are the treatments for Prostate Cancer ?",
        "expected_answer": "Treatment depends on stage, grade, age, and general health. Options include watchful waiting, surgery (radical prostatectomy), radiation therapy (external or internal), and hormonal therapy. Some men receive a combination of therapies.",
        "focus_area": "Prostate Cancer"
    },
    {
        "question": "How to prevent Osteoporosis ?",
        "expected_answer": "Prevention includes adequate calcium and vitamin D intake, weight-bearing exercise, avoiding smoking and excessive alcohol, and fall prevention measures (e.g., removing tripping hazards, using grab bars, improving lighting).",
        "focus_area": "Osteoporosis"
    },
    {
        "question": "What is (are) COPD ?",
        "expected_answer": "Chronic obstructive pulmonary disease (COPD) is a progressive lung disease in which the airways of the lungs become damaged, making it hard to breathe. It includes emphysema and chronic bronchitis.",
        "focus_area": "COPD"
    },
    {
        "question": "What are the symptoms of Stroke ?",
        "expected_answer": "Sudden numbness or weakness of the face, arm, or leg (especially on one side), sudden confusion, trouble speaking or understanding, sudden trouble seeing in one or both eyes, sudden trouble walking, dizziness, loss of balance or coordination, and sudden severe headache with no known cause.",
        "focus_area": "Stroke"
    },
    {
        "question": "What is (are) Age-related Macular Degeneration ?",
        "expected_answer": "Age-related macular degeneration (AMD) is an eye disease that affects the macula, causing blurring of sharp central vision needed for activities like reading, sewing, and driving. It causes no pain.",
        "focus_area": "Age-related Macular Degeneration"
    },
    {
        "question": "How to prevent Urinary Tract Infections ?",
        "expected_answer": "Prevention tips: wipe from front to back after using the toilet, drink lots of fluids (especially water), urinate often and when the urge arises, urinate after sex, wear cotton underwear and loose-fitting clothes, and consider cranberry juice or supplements.",
        "focus_area": "Urinary Tract Infections"
    },
    {
        "question": "What is (are) Peripheral Arterial Disease (P.A.D.) ?",
        "expected_answer": "Peripheral arterial disease (P.A.D.) is a disease in which plaque builds up in the arteries that carry blood to your head, organs, and limbs. P.A.D. usually affects the arteries in the legs.",
        "focus_area": "Peripheral Arterial Disease (P.A.D.)"
    },
    {
        "question": "What are the treatments for Breast Cancer ?",
        "expected_answer": "Treatment options include surgery (lumpectomy, mastectomy), radiation therapy, chemotherapy, hormone therapy, and targeted therapy. The choice depends on cancer stage, age, and overall health.",
        "focus_area": "Breast Cancer"
    },
    {
        "question": "What is (are) Dry Mouth ?",
        "expected_answer": "Dry mouth is the feeling that there is not enough saliva in the mouth. Everyone has dry mouth once in a while -- if they are nervous, upset, under stress, or taking certain medications. But if you have dry mouth all or most of the time, see a dentist or physician.",
        "focus_area": "Dry Mouth"
    },
    {
        "question": "What causes Paget's Disease of Bone ?",
        "expected_answer": "Researchers are not sure what causes Paget's disease. Heredity may be a factor in some cases. Research suggests that a close relative of someone with Paget's disease is seven times more likely to develop the disease than someone without an affected relative. However, most people with Paget's disease do not have any relatives with the disease. Researchers think the disease also may be caused by other factors, such as a slow-acting virus.",
        "focus_area": "Paget's Disease of Bone"
    },
    {
        "question": "What is (are) Anemia ?",
        "expected_answer": "Anemia is a condition in which your blood has a lower than normal number of red blood cells. This can cause fatigue, weakness, dizziness, and shortness of breath.",
        "focus_area": "Anemia"
    }
]

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def retrieval_score_avg(chunks):
    if not chunks:
        return 0.0
    return float(sum(c.get("score", 0.0) for c in chunks) / len(chunks))

def top_chunk_score(chunks):
    if not chunks:
        return 0.0
    return float(chunks[0].get("score", 0.0))

# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(top_k=5):
    print("=" * 70)
    print("Simplified Semantic Evaluation – Medical RAG ChatBot")
    print("=" * 70)
    print(f"Output directory: {_OUTPUT_DIR}")

    # Load embedding model for semantic similarity
    print("\nLoading embedding model for evaluation...")
    sim_model = SentenceTransformer("BAAI/bge-small-en")
    sim_model.eval()

    all_results = []
    category_scores = defaultdict(lambda: {
        "semantic_sim": [], "retrieval_avg": [], "top_score": []
    })
    failures = []

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        expected = tc["expected_answer"]
        category = tc["focus_area"]

        print(f"\n[{i+1}/{len(TEST_CASES)}] Category: {category}")
        print(f"Q: {q[:80]}")

        # Retrieval
        chunks = retrieve(q, top_k=top_k)
        ravg = retrieval_score_avg(chunks)
        rtop = top_chunk_score(chunks)

        # Generation
        generated = answer(chunks, q) if chunks else "No relevant information found."

        # Semantic similarity between generated and expected
        emb_gen = sim_model.encode(generated, normalize_embeddings=True)
        emb_exp = sim_model.encode(expected, normalize_embeddings=True)
        sim = cosine_similarity(emb_gen, emb_exp)

        # Store
        result = {
            "question": q,
            "category": category,
            "generated": generated,
            "expected": expected,
            "semantic_similarity": sim,
            "retrieval_avg_score": ravg,
            "top_chunk_score": rtop,
        }
        all_results.append(result)

        # Category aggregation
        cat = category_scores[category]
        cat["semantic_sim"].append(sim)
        cat["retrieval_avg"].append(ravg)
        cat["top_score"].append(rtop)

        # Failure if semantic similarity < 0.6
        if sim < 0.6:
            failures.append({
                "question": q,
                "category": category,
                "semantic_similarity": sim,
                "generated": generated[:200],
                "expected": expected[:200]
            })

    # Global statistics
    sim_vals = [r["semantic_similarity"] for r in all_results]
    ravg_vals = [r["retrieval_avg_score"] for r in all_results]
    rtop_vals = [r["top_chunk_score"] for r in all_results]

    print("\n" + "=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)
    print(f"Semantic similarity (generated vs expected): mean={np.mean(sim_vals):.4f}, std={np.std(sim_vals):.4f}, min={np.min(sim_vals):.4f}, max={np.max(sim_vals):.4f}")
    print(f"Retrieval avg score: mean={np.mean(ravg_vals):.4f}, std={np.std(ravg_vals):.4f}")
    print(f"Top chunk score: mean={np.mean(rtop_vals):.4f}, std={np.std(rtop_vals):.4f}")

    print("\n" + "=" * 70)
    print("PER-CATEGORY PERFORMANCE")
    print("=" * 70)
    for cat, scores in category_scores.items():
        print(f"\n{cat} (n={len(scores['semantic_sim'])}):")
        print(f"  Semantic sim: {np.mean(scores['semantic_sim']):.3f} ± {np.std(scores['semantic_sim']):.3f}")
        print(f"  Retrieval avg: {np.mean(scores['retrieval_avg']):.3f}")

    print(f"\nTotal failures (semantic sim < 0.6): {len(failures)}")
    if failures:
        print("\nFirst 5 failures:")
        for f in failures[:5]:
            print(f"  Q: {f['question'][:60]}... (sim={f['semantic_similarity']:.2f})")

    # ========== PREPARE SERIALIZABLE OUTPUTS ==========
    # Global stats as native Python types
    serializable_stats = {
        "semantic_similarity": {
            "mean": float(np.mean(sim_vals)),
            "std": float(np.std(sim_vals)),
            "min": float(np.min(sim_vals)),
            "max": float(np.max(sim_vals))
        },
        "retrieval_avg_score": {
            "mean": float(np.mean(ravg_vals)),
            "std": float(np.std(ravg_vals))
        },
        "top_chunk_score": {
            "mean": float(np.mean(rtop_vals)),
            "std": float(np.std(rtop_vals))
        }
    }

    # Per-category list
    per_category_list = []
    for cat, scores in category_scores.items():
        per_category_list.append({
            "category": cat,
            "semantic_sim_mean": float(np.mean(scores["semantic_sim"])),
            "semantic_sim_std": float(np.std(scores["semantic_sim"])),
            "retrieval_avg_mean": float(np.mean(scores["retrieval_avg"])),
            "top_score_mean": float(np.mean(scores["top_score"])),
            "count": len(scores["semantic_sim"])
        })

    # Convert all_results to native types
    serializable_results = []
    for r in all_results:
        serializable_results.append({
            "question": r["question"],
            "category": r["category"],
            "generated": r["generated"],
            "expected": r["expected"],
            "semantic_similarity": float(r["semantic_similarity"]),
            "retrieval_avg_score": float(r["retrieval_avg_score"]),
            "top_chunk_score": float(r["top_chunk_score"])
        })

    # Convert failures to native types
    serializable_failures = []
    for f in failures:
        serializable_failures.append({
            "question": f["question"],
            "category": f["category"],
            "semantic_similarity": float(f["semantic_similarity"]),
            "generated": f["generated"],
            "expected": f["expected"]
        })

    # Save JSON (evaluation.json)
    json_path = os.path.join(_OUTPUT_DIR, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "global_stats": serializable_stats,
            "per_category": per_category_list,
            "failures": serializable_failures,
            "all_results": serializable_results
        }, f, indent=2, ensure_ascii=False)

    # Save CSV (evaluation.csv) – per-category summary
    csv_path = os.path.join(_OUTPUT_DIR, "evaluation.csv")
    pd.DataFrame(per_category_list).to_csv(csv_path, index=False)

    print(f"\n✅ Results saved to:\n   {json_path}\n   {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()
    run_evaluation(top_k=args.top_k)