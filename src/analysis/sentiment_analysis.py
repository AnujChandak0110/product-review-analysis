import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# === Download resources ===
nltk.download('vader_lexicon', quiet=True)

# === Sentiment Analysis Function ===
def analyze_sentiment(
    input_path="data/processed/cleaned_reviews.csv",
    output_csv="data/processed/sentiment_results.csv",
    output_plot="results/sentiment_distribution.png"
):
    """
    Perform sentiment analysis using VADER + TextBlob.
    Saves sentiment results and visualization.
    Returns sentiment dataframe.
    """
    print("🔹 Loading cleaned reviews...")
    df = pd.read_csv(input_path)

    # Ensure correct text column
    text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
    df = df.dropna(subset=[text_col])
    reviews = df[text_col].tolist()

    # Initialize analyzer
    analyzer = SentimentIntensityAnalyzer()

    print("🧠 Performing sentiment analysis...")
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

    # Classification
    def classify_sentiment(value):
        if value >= 0.05:
            return "Positive"
        elif value <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    sentiment_df["sentiment"] = sentiment_df["vader_compound"].apply(classify_sentiment)

    # Summary
    print("\n📊 Sentiment Distribution:")
    print(sentiment_df["sentiment"].value_counts())

    # Visualization
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sentiment_df["sentiment"].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title("Overall Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sentiment_df.to_csv(output_csv, index=False)

    print(f"\n✅ Sentiment analysis complete!")
    print(f"📁 Results saved to: {output_csv}")
    print(f"🖼️ Plot saved to: {output_plot}")

    return sentiment_df


# === Standalone Execution ===
if __name__ == "__main__":
    analyze_sentiment()
