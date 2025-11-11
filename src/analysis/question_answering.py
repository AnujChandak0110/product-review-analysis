import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from collections import Counter
import re
import nltk

nltk.download('punkt', quiet=True)

# === Step 1: Load cleaned reviews ===
def load_reviews(filepath="data/processed/cleaned_reviews.csv"):
    df = pd.read_csv(filepath)
    text_col = "Cleaned_Review" if "Cleaned_Review" in df.columns else "cleaned_text"
    reviews = df[text_col].dropna().tolist()
    return reviews

# === Step 2: Get top relevant reviews for the question ===
def get_top_reviews(question, reviews, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    review_vectors = vectorizer.fit_transform(reviews)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, review_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [reviews[i] for i in top_indices]

# === Step 3: Extract keywords/phrases dynamically ===
def extract_keywords(text, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [w for w, s in sorted_scores[:top_k]]

# === Step 4: Humanize the answer dynamically ===
def humanized_answer(top_reviews):
    combined_text = " ".join(top_reviews)
    combined_text = re.sub(r'\s+', ' ', combined_text)
    sentences = sent_tokenize(combined_text)

    # Extract keywords dynamically
    keywords = extract_keywords(combined_text, top_k=8)

    # Count occurrences of keywords in sentences
    sentence_map = {kw: [] for kw in keywords}
    for kw in keywords:
        for sent in sentences:
            if re.search(r'\b' + re.escape(kw) + r'\b', sent, flags=re.IGNORECASE):
                sentence_map[kw].append(sent)

    # Form human-readable sentences
    answer_sentences = []
    for kw, sents in sentence_map.items():
        if not sents:
            continue
        # Determine sentiment for this keyword
        combined_sent = " ".join(sents)
        polarity = TextBlob(combined_sent).sentiment.polarity
        sentiment_word = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        # Pick the most common supporting phrase
        most_common_phrase = Counter(sents).most_common(1)[0][0]
        answer_sentences.append(
            f"Regarding '{kw}', users have a {sentiment_word} opinion, mentioning: \"{most_common_phrase}\""
        )

    if not answer_sentences:
        # Fallback if no keywords matched
        polarity = TextBlob(combined_text).sentiment.polarity
        sentiment_word = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        answer_sentences.append(f"Overall, users have a {sentiment_word} opinion about this product.")

    # Combine into single paragraph
    answer = " ".join(answer_sentences)
    return answer

# === Step 5: Main QA function ===
def answer_question(question, reviews=None):
    try:
        if reviews is None:
            reviews = load_reviews()
        top_reviews = get_top_reviews(question, reviews, top_n=5)
        answer = humanized_answer(top_reviews)
        return answer
    except Exception as e:
        return f"Error in QA: {e}"

# === standalone test ===
if __name__ == "__main__":
    reviews = load_reviews()
    q = input("Enter your question: ")
    print("\nAnswer:\n")
    print(answer_question(q, reviews))
