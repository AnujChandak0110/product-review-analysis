import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk import word_tokenize, pos_tag
from collections import Counter
import os

# === Setup ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

print("Loading cleaned reviews...")
df = pd.read_csv(data_path)
text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
reviews = df[text_col].dropna().tolist()

# === Build TF-IDF Matrix ===
print("Building TF-IDF matrix for reviews...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)

# === Utility: Generate short answer from top reviews ===
def summarize_reviews(relevant_reviews, aspect=None):
    words = []
    for r in relevant_reviews:
        tokens = word_tokenize(r)
        tags = pos_tag(tokens)
        words.extend([w.lower() for w, t in tags if t.startswith("JJ") or t.startswith("NN")])
    
    common = Counter(words).most_common(6)
    nouns = [w for w, _ in common if pos_tag([w])[0][1].startswith("NN")]
    adjs = [w for w, _ in common if pos_tag([w])[0][1].startswith("JJ")]

    noun_part = ", ".join(nouns[:2]) if nouns else "the product"
    adj_part = ", ".join(adjs[:2]) if adjs else "good"

    if aspect:
        return f"Customers often describe the {aspect} as {adj_part}, frequently mentioning {noun_part}."
    else:
        return f"Customers often mention {noun_part} and describe it as {adj_part}."

# === Core QA Function ===
def answer_question(question, top_n=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_reviews = [reviews[i] for i in top_indices]

    print(f"\nQuestion: {question}")
    print(f"\nMost Relevant Reviews:")
    for i, review in enumerate(top_reviews, 1):
        print(f"{i}. {review[:200]}...")

    # Generate short heuristic "answer"
    answer_summary = summarize_reviews(top_reviews)
    print(f"Answer Summary: {answer_summary}\n")

# === Interactive Mode ===
print("\nClassical QA System (TF-IDF + Similarity)")
print("Type your question (or 'exit' to quit):\n")

while True:
    user_q = input("Your question: ").strip()
    if user_q.lower() in ['exit', 'quit', 'q']:
        print("Exiting QA system.")
        break
    answer_question(user_q)
