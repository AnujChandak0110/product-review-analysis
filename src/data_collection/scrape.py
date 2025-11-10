import requests
from bs4 import BeautifulSoup
import time
import csv

def scrape_reviews(product_url, max_pages=5):
    reviews = []
    for page in range(1, max_pages+1):
        url = f"{product_url}&page={page}"
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            break
        soup = BeautifulSoup(res.text, "html.parser")
        for review in soup.select(".review"):  # update selector as per site
            text = review.get_text(strip=True)
            reviews.append({"review": text})
        time.sleep(1)
    return reviews

def save_reviews_csv(reviews, path="data/raw/reviews_raw.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review"])
        writer.writeheader()
        writer.writerows(reviews)
