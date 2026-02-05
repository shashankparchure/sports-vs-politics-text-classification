import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocessing.text_cleaning import clean_text
from features.feature_rep import (build_vocabulary, compute_idf, tfidf_features, bow_features)

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

print("\nBoW + SVM Results: \n")

X_train_bow = bow_features(X_train_texts, vocab, ngram_range=NGRAM_RANGE)
X_test_bow = bow_features(X_test_texts, vocab, ngram_range=NGRAM_RANGE)

#SVM Training
svm_bow = LinearSVC()
svm_bow.fit(X_train_bow, y_train)

#Evaluation
y_pred_bow = svm_bow.predict(X_test_bow)

print("BoW Test Accuracy:", accuracy_score(y_test, y_pred_bow))
print("\nBoW Classification Report:\n")
print(classification_report(y_test, y_pred_bow, target_names=["Politics", "Sports"]))


print("\nTF-IDF + SVM Results: \n")

idf = compute_idf(X_train_texts, vocab, ngram_range=NGRAM_RANGE)

X_train_tfidf = tfidf_features(X_train_texts, vocab, idf, ngram_range=NGRAM_RANGE)
X_test_tfidf = tfidf_features(X_test_texts, vocab, idf, ngram_range=NGRAM_RANGE)

#SVM training
svm_tfidf = LinearSVC()
svm_tfidf.fit(X_train_tfidf, y_train)

#Evaluations
y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)

print("TF-IDF Test Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("\nTF-IDF Classification Report:\n")
print(classification_report(y_test, y_pred_tfidf, target_names=["Politics", "Sports"]))


# model Saving
with open("models/svm_bow.pkl", "wb") as f:
    pickle.dump(svm_bow, f)

with open("models/svm_tfidf.pkl", "wb") as f:
    pickle.dump(svm_tfidf, f)

with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("models/idf.pkl", "wb") as f:
    pickle.dump(idf, f)

print("\nBoW and TF-IDF SVM models saved.")