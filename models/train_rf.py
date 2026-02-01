import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocessing.text_cleaning import clean_text
from features.feature_rep import (build_vocabulary, compute_idf, tfidf_features)

#Configs

DATA_PATH = "data/sports_politics_dataset.csv"
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 5000
TEST_SIZE = 0.2         # 80-20 Train-Test Split
RANDOM_STATE = 42

## Data Loading and labeling

df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["category"].tolist()

# Encoding Labels as 0-> Politics and 1-> Sports
y = np.array([1 if lbl == "Sports" else 0 for lbl in labels])



clean_texts = [clean_text(t) for t in texts]

# Train-test split

X_train_texts, X_test_texts, y_train, y_test = train_test_split(clean_texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Feature representations

vocab = build_vocabulary(X_train_texts, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)

idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

X_train = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)

X_test = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)


# Random Forest Model Training
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)

rf.fit(X_train, y_train)

# Evaluation

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Politics", "Sports"]))

with open("models/rf.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\nRandom Forest model saved.")