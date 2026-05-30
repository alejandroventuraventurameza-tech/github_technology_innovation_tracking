"""
visualization.py
----------------
Reusable chart functions for the Streamlit app and output figures.

Track B — Technology Innovation & Ecosystem Tracking
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils import LABEL2ID, ID2LABEL

LABEL_COLORS = {
    "emerging": "#2ecc71",
    "mature": "#3498db",
    "declining": "#e74c3c",
    "experimental": "#f39c12",
}


# ---------------------------------------------------------------------------
# Tab 2 — Exploratory Analysis
# ---------------------------------------------------------------------------

def plot_label_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts()
    colors = [LABEL_COLORS.get(l, "#aaa") for l in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax.set_title("Repository Label Distribution", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    return fig


def plot_signal_by_label(df: pd.DataFrame, signal: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["emerging", "mature", "declining", "experimental"]
    palette = [LABEL_COLORS[l] for l in order if l in df["label"].unique()]
    sns.boxplot(data=df, x="label", y=signal, order=order, palette=palette, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Label")
    ax.set_ylabel(signal)
    plt.tight_layout()
    return fig


def plot_activity_score_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in LABEL_COLORS.items():
        subset = df[df["label"] == label]["activity_score"]
        if len(subset) > 1:
            subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
    ax.set_title("Activity Score Distribution by Label", fontsize=13)
    ax.set_xlabel("Activity Score (0–1)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_stars_vs_age(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in LABEL_COLORS.items():
        subset = df[df["label"] == label]
        ax.scatter(
            subset["repo_age_days"] / 365,
            subset["stars"],
            label=label,
            color=color,
            alpha=0.6,
            s=30,
        )
    ax.set_xlabel("Repository Age (years)")
    ax.set_ylabel("Stars")
    ax.set_title("Stars vs. Repository Age by Label", fontsize=13)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 3 — Model Results
# ---------------------------------------------------------------------------

def plot_confusion_matrix_fig(labels, preds):
    cm = confusion_matrix(labels, preds, labels=list(LABEL2ID.values()))
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    plt.tight_layout()
    return fig


def plot_per_class_f1(metrics: dict):
    classes = list(metrics["per_class"].keys())
    f1_scores = [metrics["per_class"][c]["f1"] for c in classes]
    colors = [LABEL_COLORS.get(c, "#aaa") for c in classes]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("F1-Score per Class — Test Set", fontsize=13)
    ax.set_ylabel("F1-Score")
    ax.axhline(y=metrics["macro_f1"], color="gray", linestyle="--", linewidth=1.2, label=f"Macro F1 = {metrics['macro_f1']:.2f}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_baseline_vs_alternative():
    """Compare label distributions between baseline and refined prompts."""
    labels = ["declining", "experimental", "mature", "emerging"]
    baseline = [337, 122, 113, 15]
    refined = [258, 217, 85, 27]

    x = np.arange(len(labels))
    width = 0.35
    colors_b = "#95a5a6"
    colors_r = "#3498db"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, baseline, width, label="Baseline prompt", color=colors_b)
    ax.bar(x + width/2, refined, width, label="Refined prompt", color=colors_r)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Label Distribution: Baseline vs. Refined Prompt (Q4)", fontsize=13)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    return fig
