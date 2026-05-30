"""
evaluation.py
-------------
Evaluates the fine-tuned DistilBERT model on the held-out test set.
Produces metrics, confusion matrix, and error analysis.

Track B — Technology Innovation & Ecosystem Tracking
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from src.utils import LABEL2ID, ID2LABEL
from src.train import RepoDataset

MODEL_DIR = "models/trained_models"
TEST_PATH = "data/splits/test.csv"
OUTPUT_DIR = "output"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
METRICS_DIR = f"{OUTPUT_DIR}/metrics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LEN = 256


def load_model(model_dir: str = MODEL_DIR):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, test_df: pd.DataFrame):
    dataset = RepoDataset(test_df, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(LABEL2ID.values()), zero_division=0
    )
    macro_f1 = f1.mean()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            ID2LABEL[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(LABEL2ID))
        },
    }
    return metrics


def plot_confusion_matrix(labels, preds, output_path: str):
    cm = confusion_matrix(labels, preds, labels=list(LABEL2ID.values()))
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_metrics(metrics: dict, output_path: str):
    classes = list(metrics["per_class"].keys())
    precision = [metrics["per_class"][c]["precision"] for c in classes]
    recall = [metrics["per_class"][c]["recall"] for c in classes]
    f1 = [metrics["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    ax.bar(x, recall, width, label="Recall", color="#DD8452")
    ax.bar(x + width, f1, width, label="F1-score", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics — Test Set", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-class metrics chart saved to: {output_path}")


def error_analysis(test_df: pd.DataFrame, labels, preds, output_path: str):
    df = test_df.copy().reset_index(drop=True)
    df["true_label"] = [ID2LABEL[l] for l in labels]
    df["pred_label"] = [ID2LABEL[p] for p in preds]
    df["correct"] = df["true_label"] == df["pred_label"]

    errors = df[~df["correct"]][["full_name", "true_label", "pred_label", "summary"]]
    errors.to_csv(output_path, index=False)
    print(f"Error analysis saved to: {output_path} ({len(errors)} misclassifications)")
    return errors


def run(
    model_dir: str = MODEL_DIR,
    test_path: str = TEST_PATH,
):
    print("=" * 60)
    print("Evaluation — Test Set")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    model, tokenizer = load_model(model_dir)
    test_df = pd.read_csv(test_path)
    print(f"Test samples: {len(test_df)}")

    labels, preds = predict(model, tokenizer, test_df)

    metrics = compute_metrics(labels, preds)
    print(f"\nAccuracy : {metrics['accuracy']}")
    print(f"Macro F1 : {metrics['macro_f1']}")
    print(f"\n{classification_report(labels, preds, target_names=list(ID2LABEL.values()), zero_division=0)}")

    with open(f"{METRICS_DIR}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_DIR}/test_metrics.json")

    plot_confusion_matrix(labels, preds, f"{FIGURES_DIR}/confusion_matrix.png")
    plot_per_class_metrics(metrics, f"{FIGURES_DIR}/per_class_metrics.png")
    error_analysis(test_df, labels, preds, f"{TABLES_DIR}/errors.csv")

    return metrics


if __name__ == "__main__":
    run()
