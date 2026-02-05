import pickle
import numpy as np
import torch
import torch.nn as nn
import os

from preprocessing.text_cleaning import clean_text
from features.feature_rep import tfidf_features, bow_features

# configs

MODEL_DIR = "models/"
NGRAM_RANGE = (1, 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# features loading

with open(MODEL_DIR + "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(MODEL_DIR + "idf.pkl", "rb") as f:
    idf = pickle.load(f)

# Load models

with open(MODEL_DIR + "svm_bow.pkl", "rb") as f:
    svm_bow = pickle.load(f)

with open(MODEL_DIR + "svm_tfidf.pkl", "rb") as f:
    svm_tfidf = pickle.load(f)

with open(MODEL_DIR + "rf_bow.pkl", "rb") as f:
    rf_bow = pickle.load(f)

with open(MODEL_DIR + "rf_tfidf.pkl", "rb") as f:
    rf_tfidf = pickle.load(f)


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

ann_bow = ANN(input_size=len(vocab)).to(DEVICE)
ann_bow.load_state_dict(torch.load(MODEL_DIR + "ann_bow.pt", map_location=DEVICE))
ann_bow.eval()

ann_tfidf = ANN(input_size=len(vocab)).to(DEVICE)
ann_tfidf.load_state_dict(torch.load(MODEL_DIR + "ann_tfidf.pt", map_location=DEVICE))
ann_tfidf.eval()

# Prediction functions

def predict_text(text):
    cleaned = clean_text(text)

    X_bow = bow_features([cleaned], vocab, ngram_range=NGRAM_RANGE)
    X_tfidf = tfidf_features([cleaned], vocab, idf, ngram_range=NGRAM_RANGE)

    results = {}

    # SVM
    results["SVM + BoW"] = svm_bow.predict(X_bow)[0]
    results["SVM + TF-IDF"] = svm_tfidf.predict(X_tfidf)[0]

    # Random Forests
    results["RF + BoW"] = rf_bow.predict(X_bow)[0]
    results["RF + TF-IDF"] = rf_tfidf.predict(X_tfidf)[0]

    # ANN
    X_bow_t = torch.tensor(X_bow, dtype=torch.float32).to(DEVICE)
    X_tfidf_t = torch.tensor(X_tfidf, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        results["ANN + BoW"] = int(torch.sigmoid(ann_bow(X_bow_t)).item() > 0.5)
        results["ANN + TF-IDF"] = int(torch.sigmoid(ann_tfidf(X_tfidf_t)).item() > 0.5)

    def decode(y):
        return "Sports" if y == 1 else "Politics"

    return {k: decode(v) for k, v in results.items()}


if __name__ == "__main__":
    filename = input("Enter input text file name (.txt only): ").strip()

    if not filename.lower().endswith(".txt"):
        print("Error: Please provide a .txt file only.")
        exit(1)

    if not os.path.exists(filename):
        print("Error: File does not exist.")
        exit(1)

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("Error: Input file is empty.")
        exit(1)

    results = predict_text(text)

    print("\nPrediction Results:\n")
    for model, label in results.items():
        print(f"{model:15s} → {label}")
