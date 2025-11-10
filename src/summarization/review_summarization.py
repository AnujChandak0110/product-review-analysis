import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import os

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")
output_path = os.path.join(BASE_DIR, "data", "processed", "summary_reviews.csv")

print("Loading cleaned reviews...")
df = pd.read_csv(data_path)
text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"

# Keep only rows with non-empty reviews
df_valid = df[df[text_col].notna()].copy()
reviews = df_valid[text_col].tolist()

# === TF-IDF Representation ===
print("Converting reviews to TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)

# === Clustering Similar Reviews ===
n_clusters = 5  # number of summary clusters
print(f"Clustering {len(reviews)} reviews into {n_clusters} groups...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

# Assign cluster labels to only the valid reviews
df_valid["Cluster"] = kmeans.labels_

# === Find Representative Review (closest to centroid) ===
print("Extracting representative reviews for each cluster...")
summary_reviews = []

for i in range(n_clusters):
    cluster_indices = df_valid[df_valid["Cluster"] == i].index
    cluster_vectors = tfidf_matrix[df_valid.index.get_indexer(cluster_indices)]
    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
    similarities = cosine_similarity(cluster_vectors, centroid).flatten()
    best_idx = cluster_indices[np.argmax(similarities)]
    summary_reviews.append(df_valid.loc[best_idx, text_col])

# === Save Results ===
summary_df = pd.DataFrame({"Representative_Review": summary_reviews})
summary_df.to_csv(output_path, index=False)

print("\nReview summarization complete!")
print(f"Saved representative reviews to: {output_path}")

print("\nSummary:")
for i, review in enumerate(summary_reviews, start=1):
    print(f"\nCluster {i}:")
    print(f"→ {review[:250]}...")
