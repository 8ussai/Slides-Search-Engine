import re
import joblib

import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.metrics.pairwise import linear_kernel

from src.config import (SLIDES_CORPUS_CSV, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, DEFAULT_TOP_K, LOWERCASE_TEXT, REMOVE_PUNCTUATION, REMOVE_NUMBERS)

def clean_text(text: str) -> str:
    if LOWERCASE_TEXT:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        text = re.sub(r"[^\w\s]", " ", text)

    if REMOVE_NUMBERS:
        text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_tfidf_index():
    df = pd.read_csv(SLIDES_CORPUS_CSV)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    return df, vectorizer, tfidf_matrix

def search_tfidf(query: str, top_k: int = DEFAULT_TOP_K,) -> List[Dict]:
    df, vectorizer, tfidf_matrix = load_tfidf_index()
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    query_vec = vectorizer.transform([cleaned_query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()

    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
    results = []

    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]
        score = float(cosine_similarities[idx])
        results.append(
            {
                "rank": rank,
                "score": score,
                "doc_id": row["doc_id"],
                "page_number": int(row["page_number"]),
                "text": row["text"],
            }
        )

    return results

def interactive_cli():
    print("=== TF-IDF Search over Stanford IR Slides ===")
    print("Type your query (or just press Enter to exit).")
    print("---------------------------------------------")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            print("Exiting...")
            break

        results = search_tfidf(query, top_k=5)

        if not results:
            print("No results.")
            continue

        for res in results:
            print(f"\n[{res['rank']}] {res['doc_id']} (page {res['page_number']})")

if __name__ == "__main__":
    interactive_cli()