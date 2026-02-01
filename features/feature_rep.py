import math
import numpy as np
from collections import defaultdict

# N-Grams

def generate_ngrams(tokens, ngram_range=(1, 1)):
    ngrams = []
    min_n, max_n = ngram_range

    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i+n]))

    return ngrams


# Vocabulary building

def build_vocabulary(texts, ngram_range=(1, 1), max_features=None):
    freq = defaultdict(int)

    for text in texts:
        tokens = text.split()
        grams = generate_ngrams(tokens, ngram_range)
        for g in grams:
            freq[g] += 1

    # sorting by frequency
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    if max_features:
        sorted_items = sorted_items[:max_features]

    vocab = {word: idx for idx, (word, _) in enumerate(sorted_items)}
    return vocab


#Bag of Words(BoW)

def bow_features(texts, vocab, ngram_range=(1, 1)):
    X = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):
        tokens = text.split()
        grams = generate_ngrams(tokens, ngram_range)

        for g in grams:
            if g in vocab:
                X[i][vocab[g]] += 1

    return X


#TF - Term Frequency

def tf_features(texts, vocab, ngram_range=(1, 1)):
    X = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):
        tokens = text.split()
        grams = generate_ngrams(tokens, ngram_range)
        total = len(grams)

        if total == 0:
            continue

        counts = defaultdict(int)
        for g in grams:
            counts[g] += 1

        for g, c in counts.items():
            if g in vocab:
                X[i][vocab[g]] = c / total

    return X


#Inverse Document Frequency - IDF

def compute_idf(texts, vocab, ngram_range=(1, 1)):
    N = len(texts)
    idf = np.zeros(len(vocab))
    doc_freq = defaultdict(int)

    for text in texts:
        tokens = text.split()
        grams = set(generate_ngrams(tokens, ngram_range))

        for g in grams:
            if g in vocab:
                doc_freq[g] += 1

    for g, idx in vocab.items():
        idf[idx] = math.log((N + 1) / (doc_freq[g] + 1)) + 1

    return idf


#TF-IDF

def tfidf_features(texts, vocab, idf, ngram_range=(1, 1)):
    tf = tf_features(texts, vocab, ngram_range)
    return tf * idf
