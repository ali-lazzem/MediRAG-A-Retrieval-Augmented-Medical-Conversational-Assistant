"""Load and prepare dataset from CSV (or fallback to built-in sample)."""

import pandas as pd
from typing import List, Dict
from config import COL_QUESTION, COL_ANSWER, COL_FOCUS
import os


def load_dataset(csv_path: str = None) -> List[Dict[str, str]]:
    """
    Return a list of dicts with keys: 'question', 'answer', 'focus' (if available).
    If csv_path is None or file not found, use built-in sample.
    """
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if COL_QUESTION not in df.columns or COL_ANSWER not in df.columns:
                raise ValueError(f"CSV must contain '{COL_QUESTION}' and '{COL_ANSWER}' columns")

            records = []
            for _, row in df.iterrows():
                record = {
                    "question": str(row[COL_QUESTION]),
                    "answer": str(row[COL_ANSWER]),
                }
                if COL_FOCUS in df.columns:
                    record["focus"] = str(row[COL_FOCUS]) if pd.notna(row[COL_FOCUS]) else ""
                else:
                    record["focus"] = ""
                records.append(record)
            return records
        except Exception as e:
            print(f"Could not load CSV: {e}. Using built-in sample dataset.")
            return _get_sample_dataset()
    else:
        print("No CSV provided or file not found. Using built-in sample dataset.")
        return _get_sample_dataset()


def _get_sample_dataset() -> List[Dict[str, str]]:
    """Fallback sample medical QA pairs."""
    return [
        {
            "question": "What is glaucoma?",
            "answer": "Glaucoma is a group of eye diseases that damage the optic nerve, often caused by abnormally high pressure in the eye.",
            "focus": "Glaucoma",
        },
        {
            "question": "How can I lower high blood pressure?",
            "answer": "Lifestyle changes like reducing salt intake, exercising regularly, and taking prescribed medications can help lower high blood pressure.",
            "focus": "Hypertension",
        },
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "focus": "Diabetes",
        },
        {
            "question": "What causes Alzheimer's disease?",
            "answer": "Alzheimer's is caused by a combination of genetic, lifestyle, and environmental factors that affect the brain over time.",
            "focus": "Alzheimer",
        },
        {
            "question": "How to prevent osteoporosis?",
            "answer": "Adequate calcium and vitamin D intake, weight-bearing exercise, and avoiding smoking can help prevent osteoporosis.",
            "focus": "Osteoporosis",
        },
    ]


def get_curated_questions() -> List[str]:
    """
    Return a list of curated question strings that are less likely to be memorised
    by a pure LLM, making FAISS retrieval more beneficial. These cover diverse
    medical topics from the MedQuAD dataset.
    """
    return [
        "What are the treatments for Paget's Disease of Bone ?",
        "What is (are) Dry Mouth ?",
        "How to prevent Osteoporosis ?",
        "What is (are) Age-related Macular Degeneration ?",
        "What causes Hearing Loss ?",
        "What are the symptoms of COPD ?",
        "What is (are) Peripheral Arterial Disease (P.A.D.) ?",
        "What are the treatments for Prostate Cancer ?",
        "What is (are) Anemia ?",
        "What causes Paget's Disease of Bone ?",
        "What are the symptoms of Glaucoma ?",           # specific details, not just definition
        "How to prevent Urinary Tract Infections ?",
        "What are the treatments for Alzheimer's Disease ?",
        "What is (are) Rheumatoid Arthritis ?",
        "What causes High Blood Pressure ?",            # mechanistic causes, not just risk factors
        "What is (are) Leukemia ?",
        "What are the symptoms of Parkinson's Disease ?",
        "How to prevent Shingles ?",
        "What are the complications of Diabetic Retinopathy ?",
        "What is (are) Colorectal Cancer ?",
        "What causes Psoriasis ?",
        "How to diagnose Stroke ?",
        "What are the treatments for Heart Attack ?",
        "What is (are) Cataract ?",
        "What are the risk factors for Kidney Disease ?",
    ]


def get_curated_test_set(csv_path: str) -> List[Dict[str, str]]:
    """
    Load the full dataset from CSV and filter to only the questions in the curated list.
    Returns a list of dicts with 'question', 'answer', 'focus'.
    Raises RuntimeError if no matches are found.
    """
    full_dataset = load_dataset(csv_path)
    curated_qs = get_curated_questions()
    # Create a map with normalised keys (strip whitespace) for matching
    qa_map = {item["question"].strip(): item for item in full_dataset}
    matched = []
    missing = []
    for q in curated_qs:
        q_clean = q.strip()
        if q_clean in qa_map:
            matched.append(qa_map[q_clean])
        else:
            missing.append(q_clean)

    if missing:
        print(f"Warning: {len(missing)} curated questions not found in CSV:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more.")

    if not matched:
        raise RuntimeError("No matching questions found in CSV. Check CSV path and column names.")
    print(f"Loaded {len(matched)} test questions from curated list.")
    return matched


def build_documents(dataset: List[Dict[str, str]]) -> List[str]:
    """Construct a document (text) for each QA pair to be used for retrieval indexing."""
    docs = []
    for item in dataset:
        focus = item.get("focus", "")
        q = item["question"]
        a = item["answer"]
        if focus:
            doc = f"Focus: {focus}\nQuestion: {q}\nAnswer: {a}"
        else:
            doc = f"Question: {q}\nAnswer: {a}"
        docs.append(doc)
    return docs