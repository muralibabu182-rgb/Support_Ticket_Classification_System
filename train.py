"""
Model Training Script for Support Ticket Classification System
--------------------------------------------------------------
Trains a TF-IDF + LinearSVC pipeline on customer support ticket data,
evaluates performance metrics, and saves the serialized model.
"""

import sys
import os
import re
import string
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "all_tickets_processed_improved_v3.csv"
MODEL_PATH = "ticket_model.pkl"

def download_nltk_stopwords():
    """Ensure NLTK stopwords corpus is available."""
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return {"the", "and", "is", "in", "to", "of", "it", "for", "a", "an", "on", "at", "with"}

def clean_text(text: str, stop_words: set) -> str:
    """Preprocess raw ticket text by removing digits, punctuation, and stopwords."""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def main():
    print("=" * 60)
    print("[*] Support Ticket Classifier: Model Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"\n[1/6] Loading dataset from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Document", "Topic_group"])
    print(f"      Total valid tickets: {len(df):,}")
    print(f"      Unique categories:   {df['Topic_group'].nunique()}")

    # 2. Text Preprocessing
    print("\n[2/6] Cleaning and preprocessing ticket descriptions...")
    stop_words = download_nltk_stopwords()
    df["cleaned_text"] = df["Document"].apply(lambda t: clean_text(t, stop_words))

    X = df["cleaned_text"]
    y = df["Topic_group"].astype(str)

    # 3. Train / Test Split
    print("\n[3/6] Splitting into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train samples: {len(X_train):,}")
    print(f"      Test samples:  {len(X_test):,}")

    # 4. Pipeline Definition & Training
    print("\n[4/6] Building & Training TF-IDF + LinearSVC Pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", LinearSVC(random_state=42, dual="auto"))
    ])

    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # 5. Evaluation
    print("\n[5/6] Model Evaluation:")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      Overall Accuracy: {acc * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, digits=4))
    print("-" * 60)

    # 6. Save Model
    print(f"\n[6/6] Saving serialized pipeline to '{MODEL_PATH}'...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"      Model successfully saved to {MODEL_PATH}")
    print("\n[SUCCESS] Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
