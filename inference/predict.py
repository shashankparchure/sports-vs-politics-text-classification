import pickle
import numpy as np
import torch
import torch.nn as nn
import os

from preprocessing.text_cleaning import clean_text
from features.feature_rep import tfidf_features

# Configs

MODEL_DIR = "models/"
NGRAM_RANGE = (1, 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load feature .pkl files

with open(MODEL_DIR + "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(MODEL_DIR + "idf.pkl", "rb") as f:
    idf = pickle.load(f)

# Load Models: Random Forest, SVM and ANN

with open(MODEL_DIR + "svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open(MODEL_DIR + "rf.pkl", "rb") as f:
    rf_model = pickle.load(f)

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

ann_model = ANN(input_size=len(vocab)).to(DEVICE)
ann_model.load_state_dict(torch.load(MODEL_DIR + "ann.pt", map_location=DEVICE))
ann_model.eval()

# prediction function

def predict_text(text):
    cleaned = clean_text(text)
    X = tfidf_features([cleaned], vocab, idf, ngram_range=NGRAM_RANGE)

    svm_pred = svm_model.predict(X)[0]
    rf_pred = rf_model.predict(X)[0]

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logit = ann_model(X_tensor).item()
        ann_pred = 1 if torch.sigmoid(torch.tensor(logit)) > 0.5 else 0

    def decode(y):
        return "Sports" if y == 1 else "Politics"

    return {
        "SVM": decode(svm_pred),
        "Random Forest": decode(rf_pred),
        "ANN": decode(ann_pred)
    }


if __name__ == "__main__":
    filename = input("Enter input text file name (.txt only): ").strip()

    # ---- validation ----
    if not filename.lower().endswith(".txt"):
        print("Error: Please provide a .txt file only.")
        exit(1)

    if not os.path.exists(filename):
        print("Error: File does not exist.")
        exit(1)

    # file read
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("Error: Input file is empty.")
        exit(1)

    #result prediction
    results = predict_text(text)

    print("\nPrediction Results:\n")
    for model, label in results.items():
        print(f"{model}: {label}")