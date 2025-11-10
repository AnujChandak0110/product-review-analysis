import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import os

# === Download necessary resources ===
nltk.download('vader_lexicon', quiet=True)

# === Load Cleaned Reviews ===
print("Loading cleaned reviews...")
df = pd.read_csv("data/processed/cleaned_reviews.csv")

text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
df = df.dropna(subset=[text_col])
reviews = df[text_col].tolist()

# === Initialize VADER ===
analyzer = SentimentIntensityAnalyzer()

# === Analyze Sentiment ===
print("Performing sentiment analysis...")
sentiment_scores = []
for review in reviews:
    scores = analyzer.polarity_scores(review)
    blob = TextBlob(review)
    sentiment_scores.append({
        "review": review,
        "vader_compound": scores["compound"],
        "vader_pos": scores["pos"],
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "textblob_polarity": blob.sentiment.polarity,
    })

sentiment_df = pd.DataFrame(sentiment_scores)

# === Classify sentiment ===
def classify_sentiment(value):
    if value >= 0.05:
        return "Positive"
    elif value <= -0.05:
        return "Negative"
    else:
        return "Neutral"

sentiment_df["sentiment"] = sentiment_df["vader_compound"].apply(classify_sentiment)

# === Summary Statistics ===
print("\nSentiment Distribution:")
print(sentiment_df["sentiment"].value_counts())

# === Visualization ===
plt.figure(figsize=(6, 4))
sentiment_df["sentiment"].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

# Save to results folder
os.makedirs("results", exist_ok=True)
plt.savefig("results/sentiment_distribution.png")
plt.show()

# === Save Sentiment Data ===
os.makedirs("data/processed", exist_ok=True)
sentiment_df.to_csv("data/processed/sentiment_results.csv", index=False)

print("\nSentiment analysis complete!")
print("Results saved to: data/processed/sentiment_results.csv and results/sentiment_distribution.png")

# === Show Sample Output ===
print("\nExample Positive Reviews:")
print(sentiment_df[sentiment_df['sentiment'] == 'Positive'].sample(3)['review'].tolist())

print("\nExample Negative Reviews:")
print(sentiment_df[sentiment_df['sentiment'] == 'Negative'].sample(3)['review'].tolist())
