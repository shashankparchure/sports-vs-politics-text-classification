import csv
import sys

# Handle large text fields safely (Windows-compatible)
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int //= 10

INPUT_FILES = [
    "indian_news_with_text.csv",
    "bbc-text.csv"
]

OUTPUT_FILE = "sports_politics_dataset.csv"

def main():
    merged_rows = []

    for file in INPUT_FILES:
        with open(file, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row.get("category") in {"Sports", "Politics"} and row.get("text"):
                    merged_rows.append({
                        "category": row["category"],
                        "text": row["text"]
                    })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["category", "text"])
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Merged dataset saved as {OUTPUT_FILE}")
    print(f"Total samples: {len(merged_rows)}")

if __name__ == "__main__":
    main()
