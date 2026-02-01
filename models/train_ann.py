import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocessing.text_cleaning import clean_text
from features.feature_rep import (build_vocabulary, compute_idf, tfidf_features)

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

idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

X_train = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)

X_test = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)

# Conversion to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

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


model = ANN(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE).to(DEVICE)


criterion = nn.BCEWithLogitsLoss()  # combines Sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adam Optimizer ussed

# Training Loop

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.size(0))

    epoch_loss = 0.0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# Evaluation

model.eval()
with torch.no_grad():
    logits = model(X_test).squeeze()
    preds = torch.sigmoid(logits) > 0.5
    preds = preds.cpu().numpy().astype(int)

y_true = y_test.cpu().numpy().astype(int)

print("\nAccuracy:", accuracy_score(y_true, preds))
print("\nClassification Report:\n")
print(classification_report(y_true, preds, target_names=["Politics", "Sports"]))

torch.save(model.state_dict(), "models/ann.pt")
print("ANN model saved.")