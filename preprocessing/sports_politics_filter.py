import csv

INPUT_FILE = "india_news_10000.csv"
OUTPUT_FILE = "filtered_indian_news.csv"

QUERY_TO_CATEGORY = {
    "india politics": "Politics",
    "india elections": "Politics",
    "india cricket": "Sports"
}

def main():
    output_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            query = row.get("query", "").strip().lower()
            url = row.get("url", "").strip()

            if query in QUERY_TO_CATEGORY and url:
                output_rows.append({
                    "category": QUERY_TO_CATEGORY[query],
                    "url": url
                })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["category", "url"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows)} articles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
