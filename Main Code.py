# sentiment_system.py
# Project 2: Sentiment Analysis System (AFINN-based)
# Requires: pandas, nltk
# Optional: streamlit (for the UI app)
# Run once the first time to download NLTK punkt: it will auto-download.

import re
import os
import csv
from typing import List, Tuple, Dict, Optional
import pandas as pd

# --- Sentence splitting ---
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    import nltk
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize


# ---------------- Core utilities ----------------

def load_afinn(afinn_path: str) -> Dict[str, int]:
    """
    Load AFINN lexicon (word \t score per line) into dict.
    """
    lex = {}
    with open(afinn_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                word, score = parts
                lex[word.lower()] = int(score)
    return lex


_word_re = re.compile(r"[A-Za-z']+")

def tokenize_words(text: str) -> List[str]:
    return _word_re.findall(text.lower())


def score_sentence(sentence: str, lexicon: Dict[str, int]) -> int:
    """
    Sum of AFINN scores for tokens present in lexicon.
    """
    tokens = tokenize_words(sentence)
    return sum(lexicon.get(tok, 0) for tok in tokens)


def sentence_scores(text: str, lexicon: Dict[str, int]) -> List[Tuple[int, str, int]]:
    """
    Return list of (index, sentence, score) for each sentence in text.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    sents = sent_tokenize(text)
    out = []
    for i, s in enumerate(sents):
        out.append((i, s, score_sentence(s, lexicon)))
    return out


def sliding_window_best_segments(
    sentences: List[Tuple[int, str, int]], 
    window_size: int = 3
) -> Dict[str, Tuple[int, int, int, str]]:
    """
    Given list of (idx, sentence, score), compute sliding window sums.
    Returns dict with 'best_pos' and 'best_neg' segments:
      Each value: (start_idx, end_idx_inclusive, window_score, joined_text)
    """
    if window_size <= 0 or len(sentences) == 0:
        return {
            "best_pos": (None, None, 0, ""),
            "best_neg": (None, None, 0, "")
        }

    scores = [sc for (_, _, sc) in sentences]
    n = len(scores)
    if n < window_size:
        # Just use the whole thing if fewer than window_size sentences
        total = sum(scores)
        joined = " ".join(s for (_, s, _) in sentences)
        return {
            "best_pos": (0, n - 1, total, joined),
            "best_neg": (0, n - 1, total, joined)
        }

    # Initial window
    window_sum = sum(scores[:window_size])
    best_pos = (0, window_size - 1, window_sum)
    best_neg = (0, window_size - 1, window_sum)

    # Slide
    for start in range(1, n - window_size + 1):
        window_sum += scores[start + window_size - 1] - scores[start - 1]
        if window_sum > best_pos[2]:
            best_pos = (start, start + window_size - 1, window_sum)
        if window_sum < best_neg[2]:
            best_neg = (start, start + window_size - 1, window_sum)

    def pack(seg):
        si, ei, sc = seg
        text = " ".join(sentences[k][1] for k in range(si, ei + 1))
        return (si, ei, sc, text)

    return {"best_pos": pack(best_pos), "best_neg": pack(best_neg)}


def find_extreme_sentences(sent_triplets: List[Tuple[int, str, int]]) -> Dict[str, Tuple[int, str, int]]:
    """
    Return most positive and most negative single sentences.
    """
    if not sent_triplets:
        return {"max": (None, "", 0), "min": (None, "", 0)}
    # Break ties by earliest index
    max_item = max(sent_triplets, key=lambda x: (x[2], -x[0]))
    min_item = min(sent_triplets, key=lambda x: (x[2], x[0]))
    return {"max": max_item, "min": min_item}


# ---------------- Dataset analysis ----------------

COMMON_TEXT_COLUMNS = ["review", "text", "content", "body", "comment", "reviews"]

def detect_text_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    Try to find a text column in the dataset.
    """
    if preferred and preferred in df.columns:
        return preferred
    for c in COMMON_TEXT_COLUMNS:
        if c in df.columns:
            return c
    # If none of the common names match, pick the first object (string-like) column
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    # Fallback to the first column
    return df.columns[0]


def analyze_reviews_csv(
    csv_path: str,
    afinn_path: str,
    text_col: Optional[str] = None,
    window_size: int = 3,
    limit_rows: Optional[int] = None,
    output_dir: str = "outputs"
) -> None:
    """
    End-to-end:
      - load CSV and AFINN
      - compute per-sentence scores
      - write sentence-level and window-segment CSVs
      - print dataset-level extremes
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if limit_rows is not None:
        df = df.head(limit_rows).copy()

    lex = load_afinn(afinn_path)
    col = detect_text_column(df, preferred=text_col)

    sentence_rows = []
    segment_rows = []

    dataset_max_sentence = (None, None, "", float("-inf"))  # (row_idx, sent_idx, text, score)
    dataset_min_sentence = (None, None, "", float("inf"))

    for ridx, row in df.iterrows():
        text = str(row[col]) if pd.notna(row[col]) else ""
        sents = sentence_scores(text, lex)

        # collect sentence rows for CSV
        for (sidx, s, sc) in sents:
            sentence_rows.append({
                "row_index": ridx,
                "sentence_index": sidx,
                "sentence": s,
                "score": sc
            })
            # update dataset extremes
            if sc > dataset_max_sentence[3]:
                dataset_max_sentence = (ridx, sidx, s, sc)
            if sc < dataset_min_sentence[3]:
                dataset_min_sentence = (ridx, sidx, s, sc)

        # sliding window segments (per review)
        segs = sliding_window_best_segments(sents, window_size=window_size)
        for label in ["best_pos", "best_neg"]:
            si, ei, sc, txt = segs[label]
            segment_rows.append({
                "row_index": ridx,
                "segment_type": label,
                "start_sentence_index": si,
                "end_sentence_index": ei,
                "window_size": (ei - si + 1) if si is not None and ei is not None else 0,
                "segment_score": sc,
                "segment_text": txt
            })

    # Write outputs (Files I/O)
    sent_out = os.path.join(output_dir, "sentence_scores.csv")
    seg_out = os.path.join(output_dir, "window_segments.csv")
    pd.DataFrame(sentence_rows).to_csv(sent_out, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(segment_rows).to_csv(seg_out, index=False, quoting=csv.QUOTE_MINIMAL)

    # Print dataset-wide extremes to console (and save a quick summary)
    summary_lines = []
    summary_lines.append(f"[DATASET MOST POSITIVE SENTENCE] row={dataset_max_sentence[0]}, "
                         f"sent={dataset_max_sentence[1]}, score={dataset_max_sentence[3]}\n"
                         f"{dataset_max_sentence[2]}")
    summary_lines.append("")
    summary_lines.append(f"[DATASET MOST NEGATIVE SENTENCE] row={dataset_min_sentence[0]}, "
                         f"sent={dataset_min_sentence[1]}, score={dataset_min_sentence[3]}\n"
                         f"{dataset_min_sentence[2]}")

    print("\n".join(summary_lines))

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


if __name__ == "__main__":
    # === HOW TO RUN (example) ===
    #
    # 1) Place the AFINN file locally (download from the link in the brief) as:
    #    ./AFINN-en-165.txt
    # 2) Point csv_path to the Kaggle CSV you downloaded:
    #    e.g., "./harry_potter_reviews.csv"
    #
    # You can also override the text column name if needed, else it will auto-detect.
    #
    # Then run:  python sentiment_system.py
    #
    CSV_PATH = "./harry_potter_reviews.csv"     # <- change to your Kaggle file
    AFINN_PATH = "./AFINN-en-165.txt"           # <- change if stored elsewhere
    TEXT_COLUMN = None                           # e.g., "review" if you know the exact column
    WINDOW_SIZE = 3
    LIMIT_ROWS = None                            # e.g., 2000 to speed up dev runs

    analyze_reviews_csv(
        csv_path=CSV_PATH,
        afinn_path=AFINN_PATH,
        text_col=TEXT_COLUMN,
        window_size=WINDOW_SIZE,
        limit_rows=LIMIT_ROWS,
        output_dir="outputs"
    )


