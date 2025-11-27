from typing import List, Dict

from src.search.search_tfidf import search_tfidf
from src.search.search_embeddings import search_embeddings

def print_results(results: List[Dict], header: str, max_chars: int = 300,) -> None:
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

    if not results:
        print("No results.")
        return

    for res in results:
        rank = res.get("rank")
        score = res.get("score")
        doc_id = res.get("doc_id")
        page_number = res.get("page_number")
        text = res.get("text", "")
        snippet = text

        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."

        print(f"\n[{rank}] {doc_id} (page {page_number})  |  score = {score:.4f}")
        print(f"    {snippet}")

def choose_mode() -> str:
    print("Choose search mode:")
    print("1) TF-IDF (keyword-based IR)")
    print("2) Embeddings (semantic search)")
    print("3) Both (compare TF-IDF and Embeddings)")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    print("=== Unified IR Search over Stanford IR Slides ===")
    print("This CLI lets you search using:")
    print("  - TF-IDF (classic IR)")
    print("  - Transformer Embeddings (semantic search)")
    print("-------------------------------------------------")

    mode = choose_mode()
    print("\nType your query (press Enter on empty line to exit).")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            print("Exiting...")
            break

        if mode == "1":
            results_tfidf = search_tfidf(query, top_k=5)
            print_results(results_tfidf, header="TF-IDF Results (Top-5)")

        elif mode == "2":
            results_emb = search_embeddings(query, top_k=5)
            print_results(results_emb, header="Embeddings Semantic Search Results (Top-5)",)

        else: 
            results_tfidf = search_tfidf(query, top_k=5)
            results_emb = search_embeddings(query, top_k=5)
            print_results(results_tfidf, header="TF-IDF Results (Top-5)")
            print_results(results_emb, header="Embeddings Semantic Search Results (Top-5)",)

if __name__ == "__main__":
    main()