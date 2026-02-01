import re

STOPWORDS = {
    "the", "is", "and", "to", "of", "in", "for", "on", "with", "as",
    "at", "by", "an", "be", "this", "that", "from", "it", "are"
}

def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)     # removes URLs
    text = re.sub(r"\d+", " num ", text)     # normalizes numbers
    text = re.sub(r"[^a-z\s]", " ", text)    # keep letters + num token
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]

    return " ".join(tokens)
