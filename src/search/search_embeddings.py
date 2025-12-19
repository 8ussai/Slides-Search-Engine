import pickle
import re

import numpy as np
import pandas as pd

from typing import List, Dict
from sentence_transformers import SentenceTransformer

from src.config import (SLIDES_CORPUS_CSV, EMBEDDINGS_NPY_PATH, EMBEDDINGS_METADATA_PATH, SENTENCE_TRANSFORMER_MODEL, DEFAULT_TOP_K, LOWERCASE_TEXT, REMOVE_PUNCTUATION, REMOVE_NUMBERS)

def clean_text(text: str) -> str:
    if LOWERCASE_TEXT:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        text = re.sub(r"[^\w\s]", " ", text)

    if REMOVE_NUMBERS:
        text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

_df_cache = None
_embeddings_cache = None
_metadata_cache = None
_model_cache = None

def load_embeddings_index():
    global _df_cache, _embeddings_cache, _metadata_cache, _model_cache

    if _df_cache is None:
        _df_cache = pd.read_csv(SLIDES_CORPUS_CSV)

    if _embeddings_cache is None or _metadata_cache is None:
        _embeddings_cache = np.load(EMBEDDINGS_NPY_PATH)

        with open(EMBEDDINGS_METADATA_PATH, "rb") as f:
            _metadata_cache = pickle.load(f)

    if _model_cache is None:
        _model_cache = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    return _df_cache, _embeddings_cache, _metadata_cache, _model_cache

def cosine_similarity(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    q = query_vec.reshape(1, -1)
    sims = np.dot(doc_embeddings, q.T).flatten()

    return sims

def search_embeddings(query: str, top_k: int = DEFAULT_TOP_K,) -> List[Dict]:
    df, embeddings, metadata, model = load_embeddings_index()
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    query_vec = model.encode([cleaned_query], convert_to_numpy=True, normalize_embeddings=True,)[0]
    cosine_similarities = cosine_similarity(query_vec, embeddings)

    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc_id, page_number = metadata[idx]
        row = df.iloc[idx]
        score = float(cosine_similarities[idx])
        results.append(
            {
                "rank": rank,
                "score": score,
                "doc_id": doc_id,
                "page_number": int(page_number),
                "text": row["text"],
            }
        )
        
    return results

def interactive_cli():
    print("=== Embeddings Semantic Search over Stanford IR Slides ===")
    print("Type your query (or just press Enter to exit).")
    print("--------------------------------------------------------")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            print("Exiting...")
            break

        results = search_embeddings(query, top_k=5)

        if not results:
            print("No results.")
            continue

        for res in results:
            print(f"\n[{res['rank']}] {res['doc_id']} (page {res['page_number']})")

if __name__ == "__main__":
    interactive_cli()