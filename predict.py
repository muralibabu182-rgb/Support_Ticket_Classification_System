"""
CLI Prediction Utility for Support Ticket Classification
--------------------------------------------------------
Supports single ticket text inference, interactive prompt mode,
and batch CSV classification with confidence estimation.
"""

import sys
import os
import argparse
import re
import string
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "ticket_model.pkl"

def download_nltk_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return {"the", "and", "is", "in", "to", "of", "it", "for", "a", "an", "on", "at", "with"}

def clean_text(text: str, stop_words: set) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def softmax(x):
    """Compute softmax values for a 2D matrix of scores/logits."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model '{MODEL_PATH}' not found. Please run 'python train.py' first."
        )
    return joblib.load(MODEL_PATH)

def predict_single(text: str, model, stop_words: set):
    cleaned = clean_text(text, stop_words)
    clf = model.named_steps["clf"]
    classes = clf.classes_
    
    # Get decision function scores
    scores = model.decision_function([cleaned])
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    
    probs = softmax(scores)[0]
    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx]
    
    # Top 3 classes
    sorted_indices = np.argsort(probs)[::-1]
    top_predictions = [
        (classes[idx], probs[idx]) for idx in sorted_indices[:3]
    ]
    
    return pred_class, confidence, top_predictions

def predict_batch(csv_in: str, csv_out: str, text_col: str, model, stop_words: set):
    if not os.path.exists(csv_in):
        raise FileNotFoundError(f"Input file '{csv_in}' does not exist.")
    
    print(f"Reading tickets from '{csv_in}'...")
    df = pd.read_csv(csv_in)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"Processing {len(df):,} tickets...")
    cleaned_texts = df[text_col].apply(lambda t: clean_text(t, stop_words))
    
    clf = model.named_steps["clf"]
    classes = clf.classes_
    scores = model.decision_function(cleaned_texts)
    probs = softmax(scores)
    
    pred_indices = np.argmax(probs, axis=1)
    df["Predicted_Category"] = classes[pred_indices]
    df["Confidence"] = np.round(np.max(probs, axis=1) * 100, 2)
    
    df.to_csv(csv_out, index=False)
    print(f"[SUCCESS] Predictions successfully saved to '{csv_out}'")

def main():
    parser = argparse.ArgumentParser(
        description="Predict categories for support tickets using the trained NLP model."
    )
    parser.add_argument("text_pos", nargs="*", help="Direct ticket text to classify")
    parser.add_argument("--text", "-t", type=str, help="Ticket description text to classify")
    parser.add_argument("--input", "-i", type=str, help="Path to input CSV file for batch processing")
    parser.add_argument("--output", "-o", type=str, default="classified_tickets.csv", help="Output path for batch predictions")
    parser.add_argument("--column", "-c", type=str, default="Document", help="Text column name in input CSV")

    args = parser.parse_args()

    stop_words = download_nltk_stopwords()
    model = load_pipeline()

    # Determine input text from positional or flag
    input_text = args.text
    if not input_text and args.text_pos:
        input_text = " ".join(args.text_pos)

    if args.input:
        predict_batch(args.input, args.output, args.column, model, stop_words)
    elif input_text:
        pred_class, conf, top3 = predict_single(input_text, model, stop_words)
        print("\n" + "=" * 50)
        print(f"Input Ticket : {input_text}")
        print(f"Predicted    : {pred_class} ({conf * 100:.1f}% confidence)")
        print("-" * 50)
        print("Top 3 Predictions:")
        for rank, (cat, prob) in enumerate(top3, 1):
            bar = "#" * int(prob * 20)
            print(f"  {rank}. {cat:<22} {prob*100:>5.1f}%  {bar}")
        print("=" * 50 + "\n")
    else:
        # Interactive mode
        print("\nSupport Ticket Classifier - Interactive CLI Mode")
        print("Type a ticket description and press Enter (or type 'exit' / 'quit' to stop).\n")
        while True:
            try:
                user_ticket = input("Enter Ticket > ").strip()
                if user_ticket.lower() in ("exit", "quit", "q"):
                    print("Exiting.")
                    break
                if not user_ticket:
                    continue
                pred_class, conf, top3 = predict_single(user_ticket, model, stop_words)
                print(f"--> Predicted: {pred_class} (Confidence: {conf*100:.1f}%)")
                print(f"    Breakdown: " + ", ".join([f"{c}: {p*100:.0f}%" for c, p in top3]))
                print("-" * 40)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

if __name__ == "__main__":
    main()
