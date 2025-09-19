import csv
import re

# -------- Load AFINN Dictionary --------
def load_afinn(path):
    afinn = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, score = line.strip().split("\t")
            afinn[word] = int(score)
    return afinn

# -------- Clean + Score one sentence --------
def score_sentence(sentence, afinn):
    words = re.findall(r"[a-zA-Z]+", sentence.lower())
    score = 0
    for w in words:
        if w in afinn:
            score += afinn[w]
    return score

# -------- Analyze dataset --------
def analyze(csv_file, afinn_file):
    afinn = load_afinn(afinn_file)

    results = []
    max_sent = ("", -9999)
    min_sent = ("", 9999)

    # Use DictReader so we can pick 'comment' column by name
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            review = row["comment"]   # <-- use the review text
            sentences = re.split(r"[.!?]+", review)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                score = score_sentence(sent, afinn)
                results.append([idx, sent, score])
                if score > max_sent[1]:
                    max_sent = (sent, score)
                if score < min_sent[1]:
                    min_sent = (sent, score)

    # Save results to a NEW CSV
    with open("sentence_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_index", "sentence", "score"])
        writer.writerows(results)

    # Print summary
    print("Most Positive Sentence:", max_sent)
    print("Most Negative Sentence:", min_sent)

# -------- MAIN --------
if __name__ == "__main__":
    CSV_FILE = "harry_potter_reviews.csv"   # your Kaggle dataset file
    AFINN_FILE = "AFINN-en-165.txt"         # dictionary file
    analyze(CSV_FILE, AFINN_FILE)



