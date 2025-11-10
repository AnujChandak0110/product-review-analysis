import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

print("🔹 Loading cleaned reviews...")
df = pd.read_csv(data_path)

text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
reviews = df[text_col].dropna().tolist()

# === TF-IDF Vectorization ===
print("🔹 Creating TF-IDF model...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(reviews)
terms = vectorizer.get_feature_names_out()

# Create a mapping of word → vector index
word_vectors = {word: tfidf_matrix[:, i].toarray().flatten() for i, word in enumerate(terms)}

# === Define Key Product Features to Explore ===
target_words = ["camera", "battery", "screen", "performance", "price"]

def find_similar_words(word, top_n=5):
    if word not in word_vectors:
        return []
    similarities = {}
    target_vec = word_vectors[word]
    for w, vec in word_vectors.items():
        if w == word:
            continue
        sim = cosine_similarity([target_vec], [vec])[0][0]
        similarities[w] = sim
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

# === Compute Similar Words ===
results = []
print("\n🔹 Computing word similarities...\n")
for target in target_words:
    similar = find_similar_words(target)
    if similar:
        print(f"{target.upper()} → {[w for w, _ in similar]}")
        for w, score in similar:
            results.append({"Target": target, "Similar_Word": w, "Cosine_Similarity": round(score, 4)})
    else:
        print(f"{target.upper()} → Not found in vocabulary.")

# === Save Results ===
output_path = os.path.join(BASE_DIR, "data", "processed", "word_similarity.csv")
pd.DataFrame(results).to_csv(output_path, index=False)

print(f"\nSaved word similarity results to: {output_path}")
