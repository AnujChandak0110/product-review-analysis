# 🧠 Comprehensive Product Review Analysis (Classical NLP Approach)

This project performs **end-to-end NLP analysis** on customer reviews using **traditional NLP techniques** (no Transformers).  
It demonstrates how classical methods can extract meaningful insights, sentiment, and summaries from real-world product reviews.

---

## 🎯 Project Objectives
- Extract and clean customer reviews (from Flipkart)
- Translate multilingual text to English
- Perform linguistic analysis (POS tagging, NER)
- Conduct sentiment analysis (lexicon-based)
- Identify semantic relations between key features
- Summarize reviews and answer user queries

---

## 🏗️ Steps Implemented
1. **Web Scraping** – Collected product reviews using `BeautifulSoup`
2. **Preprocessing & Translation** – Cleaning, tokenization, and language translation
3. **POS & NER Analysis** – Extracting product-related entities and adjectives
4. **Sentiment Analysis** – Using polarity scoring and visualization
5. **Vector Semantics** – Word similarity via TF-IDF & cosine similarity
6. **Summarization** – Generating concise review summaries
7. **Interactive QA System** – Answering user questions based on review data

---

## 🧰 Tech Stack
**Languages & Libraries:**
- Python  
- BeautifulSoup, Regex  
- NLTK, Spacy, TextBlob  
- Scikit-learn, Gensim  
- Matplotlib, Pandas  
- Googletrans (for translation)

---

## 📁 Folder Structure
product_review_analysis/
├── data/
│ ├── raw/ # Original scraped data
│ ├── processed/ # Cleaned and translated reviews
│ └── results/ # Output visualizations and CSVs
│
├── src/
│ ├── preprocessing/ # Cleaning & translation scripts
│ ├── analysis/ # POS, sentiment, vector & QA scripts
│ └── summarization/ # Review summarization scripts
│
├── main.py # Pipeline entry point
├── requirements.txt
└── README.md

## 🚀 How to Run
1. Clone this repository  
   git clone 
   cd product_review_analysis

2. Install dependencies
   pip install -r requirements.txt

3. Run the main pipeline
   python main.py

4. Choose the desired phase (Preprocessing → Sentiment → QA → Summarization)

## 📊 Outputs
- Sentiment distribution plots
- POS & entity statistics
- Top similar words for key features
- Review summaries
- Answers to user queries

## ✨ Author

Developed by **Anuj Chandak**
NLP Mini Project — **Product Review Analysis**