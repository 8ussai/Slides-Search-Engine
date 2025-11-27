import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (ensure_directories, PROCESSED_DIR, SLIDES_CORPUS_CSV, TFIDF_DIR, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF)

def build_tfidf_index():
    ensure_directories()

    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SLIDES_CORPUS_CSV)
    texts = df["text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

if __name__ == "__main__":
    build_tfidf_index()