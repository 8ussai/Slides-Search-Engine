import csv
import re
import pdfplumber

from pathlib import Path

from src.config import (ensure_directories, RAW_SLIDES_DIR, PROCESSED_DIR, SLIDES_CORPUS_CSV, LOWERCASE_TEXT, REMOVE_PUNCTUATION, REMOVE_NUMBERS,)

def clean_text(text: str) -> str:
    if LOWERCASE_TEXT:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        text = re.sub(r"[^\w\s]", " ", text)

    if REMOVE_NUMBERS:
        text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def extract_text_from_pdf(pdf_path: Path):
    rows = []
    doc_id = pdf_path.stem  

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text:
                continue

            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                continue

            rows.append(
                {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "text": cleaned_text,
                }
            )

    return rows

def build_slides_corpus():
    ensure_directories()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_SLIDES_DIR.glob("*.pdf"))
    all_rows = []

    for pdf_path in pdf_files:
        rows = extract_text_from_pdf(pdf_path)
        all_rows.extend(rows)

    with open(SLIDES_CORPUS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "page_number", "text"])
        writer.writeheader()
        writer.writerows(all_rows)

if __name__ == "__main__":
    build_slides_corpus()