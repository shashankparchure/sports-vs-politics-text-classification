import csv
import sys

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int //= 10

FILE_PATH = "indian_news_with_text.csv"

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = [
            {"category": row["category"], "text": row["text"]}
            for row in reader
            if "category" in row and "text" in row
        ]

    with open(FILE_PATH, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["category", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Rows retained: {len(rows)}")

if __name__ == "__main__":
    main()
