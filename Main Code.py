import pandas as pd

# --- STEP 1: Load AFINN dictionary ---
def load_afinn(file_path):
    afinn = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word, score = line.strip().split("\t")
            afinn[word] = int(score)
    return afinn


# --- STEP 2: Calculate sentence score ---
def score_sentence(sentence, afinn):
    words = sentence.lower().split()
    score = 0
    for w in words:
        if w in afinn:
            score += afinn[w]
    return score


# --- STEP 3: Analyze all reviews ---
def analyze_reviews(csv_file, afinn_file):
    # load files
    df = pd.read_csv(csv_file)
    afinn = load_afinn(afinn_file)

    # pick the column with reviews (auto-detect or edit manually)
    if "review" in df.columns:
        text_col = "review"
    else:
        text_col = df.columns[0]   # just use first column

    all_scores = []   # store results
    max_sentence = ("", -9999)  # (text, score)
    min_sentence = ("", 9999)

    # go through each review
    for idx, review in enumerate(df[text_col].dropna()):
        sentences = review.split(".")   # simple split by "."
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            score = score_sentence(sent, afinn)
            all_scores.append([idx, sent, score])

            # update extremes
            if score > max_sentence[1]:
                max_sentence = (sent, score)
            if score < min_sentence[1]:
                min_sentence = (sent, score)

    # save results into CSV (Files I/O)
    result_df = pd.DataFrame(all_scores, columns=["review_index", "sentence", "score"])
    result_df.to_csv("sentence_scores.csv", index=False)

    # print summary
    print("Most Positive Sentence:", max_sentence)
    print("Most Negative Sentence:", min_sentence)


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    CSV_PATH = "harry_potter_reviews.csv"   # your dataset file
    AFINN_PATH = "AFINN-en-165.txt"         # the dictionary file

    analyze_reviews(CSV_PATH, AFINN_PATH)


