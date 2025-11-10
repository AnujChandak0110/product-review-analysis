import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_dir = os.path.join(BASE_DIR, "data", "processed")
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# === LOAD DATASETS ===
sentiment_path = os.path.join(processed_dir, "sentiment_results.csv")
topic_path = os.path.join(processed_dir, "lsa_topics.csv")
similarity_path = os.path.join(processed_dir, "word_similarity.csv")

print("Loading data for visualization...")

sentiments = pd.read_csv(sentiment_path)
topics = pd.read_csv(topic_path)
similarity = pd.read_csv(similarity_path)

# === SENTIMENT DISTRIBUTION ===
plt.figure(figsize=(6, 4))
sns.countplot(x="sentiment", data=sentiments, palette="coolwarm", hue="sentiment", legend=False)
plt.title("Sentiment Distribution of Reviews")
plt.xlabel("Sentiment Category")
plt.ylabel("Review Count")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "sentiment_distribution.png"))
plt.close()

# === TOP TOPICS (from LSA) ===
plt.figure(figsize=(8, 4))

# Combine all Topic columns into a single Series using pd.concat
topic_columns = [col for col in topics.columns if col.lower().startswith("topic")]
all_topics = pd.concat([topics[col].astype(str) for col in topic_columns], ignore_index=True)

topic_counts = all_topics.value_counts().head(5)

sns.barplot(x=topic_counts.index, y=topic_counts.values, palette="viridis")
plt.title("Top 5 Topics in Reviews (LSA)")
plt.xlabel("Topic ID")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "top_topics.png"))
plt.close()

# === WORD SIMILARITY HEATMAP ===
if not similarity.empty:
    pivot_table = similarity.pivot(index="Target", columns="Similar_Word", values="Cosine_Similarity")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Semantic Similarity between Product Features and Related Words")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "word_similarity_heatmap.png"))
    plt.close()

# === COMBINE REPORT SUMMARY ===
print("\nVisualizations saved in 'results/' folder:")
for file in os.listdir(results_dir):
    print("  -", file)
