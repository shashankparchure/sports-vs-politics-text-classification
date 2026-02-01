import csv
import time
import requests
from bs4 import BeautifulSoup

INPUT_FILE = "filtered_indian_news.csv"
OUTPUT_FILE = "indian_news_with_text.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def extract_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        text = text.replace("\n", " ").strip()
        return text if len(text) > 200 else None

    except Exception:
        return None

def main():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            url = row["url"]
            category = row["category"]

            article_text = extract_text(url)

            if article_text:
                rows.append({
                    "category": category,
                    "url": url,
                    "text": article_text
                })

            time.sleep(1)  # be polite to servers

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["category", "url", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} articles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()