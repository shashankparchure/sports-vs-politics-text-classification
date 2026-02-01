import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocessing.text_cleaning import clean_text
from features.feature_rep import (build_vocabulary, compute_idf, tfidf_features)

#CONFIGS

DATA_PATH = "data/sports_politics_dataset.csv"
NGRAM_RANGE = (1, 2)          # (1,1) for unigram, (1,2) for uni+bi -gram
MAX_FEATURES = 5000
TEST_SIZE = 0.2               # 80-20 Train-Test Split
RANDOM_STATE = 42

# Data Loading and labeliing

df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
labels = df["category"].tolist()

# Encode labels
# Since we have only 2 labels in this assignment, therefore, 0-> Politics and 1-> Sports
y = np.array([1 if lbl == "Sports" else 0 for lbl in labels])

# preprocessing and data cleaning
clean_texts = [clean_text(t) for t in texts]

# Train-Test Split

X_train_texts, X_test_texts, y_train, y_test = train_test_split(clean_texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Feature representations

vocab = build_vocabulary(X_train_texts, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)

idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

X_train = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)

X_test = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)


#SVM Training
svm = LinearSVC()
svm.fit(X_train, y_train)

#Evaluations

y_pred = svm.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Politics", "Sports"]))


#model saving
with open("models/svm.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("models/idf.pkl", "wb") as f:
    pickle.dump(idf, f)

print("\nSVM model and feature representations saved.")