import pickle

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.config import (ensure_directories, PROCESSED_DIR, SLIDES_CORPUS_CSV, EMBEDDINGS_DIR, EMBEDDINGS_NPY_PATH, EMBEDDINGS_METADATA_PATH, SENTENCE_TRANSFORMER_MODEL)

def build_embeddings_index():
    ensure_directories()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SLIDES_CORPUS_CSV)
    texts = df["text"].astype(str).tolist()
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    np.save(EMBEDDINGS_NPY_PATH, embeddings)

    metadata = list(
        zip(df["doc_id"].tolist(), df["page_number"].astype(int).tolist())
    )
    with open(EMBEDDINGS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    build_embeddings_index()