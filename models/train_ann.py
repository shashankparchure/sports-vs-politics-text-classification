import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocessing.text_cleaning import clean_text
from features.feature_rep import (build_vocabulary, compute_idf, tfidf_features, bow_features)

# Configs

DATA_PATH = "data/sports_politics_dataset.csv"
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 5000

HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading and labeling

df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["category"].tolist()

# Binary labels: Sports=1, Politics=0
y = np.array([1 if lbl == "Sports" else 0 for lbl in labels])

# data cleaning and preprocessing

clean_texts = [clean_text(t) for t in texts]

X_train_texts, X_test_texts, y_train, y_test = train_test_split(clean_texts, y, test_size=0.2, random_state=42, stratify=y)

# Feature representations

vocab = build_vocabulary(X_train_texts, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)

# idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

# X_train = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)

# X_test = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)

# # Conversion to torch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
# y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

# X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
# y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# ANN Model
#Here I have used a ML model. It has 2 hidden layers. After trying out several activation function combinations, ReLU and tanh functions are used.
#More details in report.

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)                  # Raw logits
        return x

print("\nBoW + ANN Results: \n")

X_train_bow = bow_features(X_train_texts, vocab, ngram_range=NGRAM_RANGE)
X_test_bow = bow_features(X_test_texts, vocab, ngram_range=NGRAM_RANGE)

X_train_bow = torch.tensor(X_train_bow, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

X_test_bow = torch.tensor(X_test_bow, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

model_bow = ANN(input_size=X_train_bow.shape[1], hidden_size=HIDDEN_SIZE).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()  # combines Sigmoid + BCE
optimizer = optim.Adam(model_bow.parameters(), lr=LEARNING_RATE) #Adam Optimizer ussed

# Training Loop

for epoch in range(EPOCHS):
    model_bow.train()
    perm = torch.randperm(X_train_bow.size(0))

    epoch_loss = 0.0

    progress_bar = tqdm(
        range(0, X_train_bow.size(0), BATCH_SIZE),
        desc=f"BoW Epoch {epoch+1}/{EPOCHS}",
        leave=False
    )

    for i in progress_bar:
        idx = perm[i:i+BATCH_SIZE]
        batch_x = X_train_bow[idx]
        batch_y = y_train_t[idx]

        optimizer.zero_grad()
        outputs = model_bow(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())


print("BoW Training Complete")

# Evaluation

model_bow.eval()
with torch.no_grad():
    logits = model_bow(X_test_bow).squeeze()
    preds = torch.sigmoid(logits) > 0.5
    preds = preds.cpu().numpy().astype(int)

print("\nBoW Accuracy:", accuracy_score(y_test, preds))
print("\nBoW Classification Report:\n")
print(classification_report(y_test, preds, target_names=["Politics", "Sports"]))


print("\nTF-IDF + ANN Results: \n")

idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

X_train_tfidf = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)
X_test_tfidf = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)

X_train_tfidf = torch.tensor(X_train_tfidf, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

X_test_tfidf = torch.tensor(X_test_tfidf, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

model_tfidf = ANN(input_size=X_train_tfidf.shape[1], hidden_size=HIDDEN_SIZE).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_tfidf.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model_tfidf.train()
    perm = torch.randperm(X_train_tfidf.size(0))

    epoch_loss = 0.0

    progress_bar = tqdm(
        range(0, X_train_tfidf.size(0), BATCH_SIZE),
        desc=f"TF-IDF Epoch {epoch+1}/{EPOCHS}",
        leave=False
    )

    for i in progress_bar:
        idx = perm[i:i+BATCH_SIZE]
        batch_x = X_train_tfidf[idx]
        batch_y = y_train_t[idx]

        optimizer.zero_grad()
        outputs = model_tfidf(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())



print("TF-IDF Training Complete")

model_tfidf.eval()
with torch.no_grad():
    logits = model_tfidf(X_test_tfidf).squeeze()
    preds = torch.sigmoid(logits) > 0.5
    preds = preds.cpu().numpy().astype(int)

print("\nTF-IDF Accuracy:", accuracy_score(y_test, preds))
print("\nTF-IDF Classification Report:\n")
print(classification_report(y_test, preds, target_names=["Politics", "Sports"]))


# Model Saving
torch.save(model_bow.state_dict(), "models/ann_bow.pt")
torch.save(model_tfidf.state_dict(), "models/ann_tfidf.pt")

print("BoW and TF-IDF ANN models saved.")
