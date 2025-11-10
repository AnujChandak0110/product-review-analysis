from src.data_collection.scrape import scrape_reviews, save_reviews_csv
from src.preprocessing.text_cleaning import clean_text
import pandas as pd

def main():
    url = "https://example.com/product?pid=XXXX"  # replace later
    reviews = scrape_reviews(url, max_pages=5)
    save_reviews_csv(reviews)

    df = pd.DataFrame(reviews)
    print(df.head())
    df["cleaned"] = df["review"].apply(clean_text)
    df.to_csv("data/processed/reviews_cleaned.csv", index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
