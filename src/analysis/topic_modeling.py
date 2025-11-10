import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# === Project Root Path ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === Load Cleaned Reviews ===
print("Loading cleaned reviews...")
df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv"))

text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
df = df.dropna(subset=[text_col])
reviews = df[text_col].tolist()

# === TF-IDF Vectorization ===
print("Creating TF-IDF matrix...")
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)

# === LSA (TruncatedSVD) ===
print("Performing Topic Modeling (LSA)...")
n_topics = 5  # number of topics you want to extract
lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

# === Top Keywords per Topic ===
terms = vectorizer.get_feature_names_out()
topics = []
print("\n=== Top Words per Topic ===")
for i, comp in enumerate(lsa_model.components_):
    terms_in_topic = [terms[idx] for idx in comp.argsort()[-10:][::-1]]
    print(f"Topic {i+1}: {', '.join(terms_in_topic)}")
    topics.append(terms_in_topic)

# === Save Topic Results ===
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
topics_df = pd.DataFrame(topics).transpose()
topics_df.columns = [f"Topic_{i+1}" for i in range(n_topics)]
topics_path = os.path.join(BASE_DIR, "data", "processed", "lsa_topics.csv")
topics_df.to_csv(topics_path, index=False)

print(f"\nTopics saved to: {topics_path}")
