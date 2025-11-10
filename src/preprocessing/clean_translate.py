"""
clean_translate.py
------------------
Performs:
1. Language detection
2. Translation (Hindi → English)
3. Data cleaning (HTML tags, special chars, duplicates)
4. Normalization (lowercasing, tokenization, stopword removal, lemmatization)

"""

import os
import re
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from bs4 import BeautifulSoup
import nltk

# Download required NLTK data (first run only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# === Paths ===
RAW_FILE = os.path.join("data", "raw", "infinix_reviews.csv")
OUTPUT_FILE = os.path.join("data", "processed", "cleaned_reviews.csv")

# === Initialize tools ===
translator = GoogleTranslator(source='auto', target='en')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# === Helper Functions ===

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text, lang):
    if lang == "hi":  # Hindi
        try:
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print("⚠️ Translation failed:", e)
            return text
    else:
        return text

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    # Normalize case
    text = text.lower().strip()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# === Pipeline ===

def preprocess_reviews():
    print("Loading raw reviews...")
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} reviews.")

    cleaned_reviews = []
    for idx, row in df.iterrows():
        review = str(row["Review"])
        if not isinstance(review, str) or review.strip() == "":
            continue

        lang = detect_language(review)
        translated = translate_to_english(review, lang)
        cleaned = clean_text(translated)
        normalized = preprocess_text(cleaned)

        cleaned_reviews.append({
            "Rating": row.get("Rating", ""),
            "Title": row.get("Title", ""),
            "Original_Review": review,
            "Language": lang,
            "Translated_Review": translated,
            "Cleaned_Review": normalized
        })

        if idx % 20 == 0:
            print(f"Processed {idx} reviews...")

    df_cleaned = pd.DataFrame(cleaned_reviews)
    df_cleaned.drop_duplicates(subset="Cleaned_Review", inplace=True)
    df_cleaned.dropna(subset=["Cleaned_Review"], inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPreprocessing complete!")
    print(f"Saved cleaned data to {OUTPUT_FILE}")
    print(f"Final reviews count: {len(df_cleaned)}")

if __name__ == "__main__":
    preprocess_reviews()
