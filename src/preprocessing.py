"""
preprocessing.py
----------------
Cleans and transforms raw GitHub repository data collected by
github_collector.py. Handles missing values, outliers, feature
engineering, and prepares a clean dataset ready for summarization
and modeling.

Track B — Technology Innovation & Ecosystem Tracking
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import LABEL2ID

RAW_PATH = "data/raw/repositories.csv"
PROCESSED_PATH = "data/processed/repositories_clean.csv"
SUMMARIZED_PATH = "data/processed/repositories_summarized.csv"
LABELED_PATH = "data/labeled/repositories_labeled.csv"

SPLITS_DIR = "data/splits"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} repositories from {path}")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a clean DataFrame."""

    df = df.copy()

    # --- 1. Drop exact duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset="full_name")
    print(f"Duplicates removed: {before - len(df)}")

    # --- 2. Fill missing text fields ---
    df["description"] = df["description"].fillna("").str.strip()
    df["language"] = df["language"].fillna("unknown")
    df["topics"] = df["topics"].fillna("")
    df["license"] = df["license"].fillna("none")

    # --- 3. Fill missing numeric signals with 0 ---
    numeric_cols = [
        "stars", "forks", "open_issues", "watchers",
        "contributors_count", "weekly_commit_avg", "releases_count",
        "closed_issues_count", "readme_length",
        "repo_age_days", "days_since_last_push",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- 4. Fix boolean column ---
    df["has_ci"] = df["has_ci"].astype(str).str.lower().isin(["true", "1", "yes"])

    # --- 5. Cap extreme outliers using 99th percentile ---
    for col in ["stars", "forks", "contributors_count", "readme_length"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # --- 6. Parse dates ---
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["pushed_at"] = pd.to_datetime(df["pushed_at"], utc=True, errors="coerce")

    print(f"Clean dataset: {len(df)} repositories")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features useful for classification."""

    df = df.copy()

    # Issue resolution rate — proxy for community responsiveness
    total_issues = df["open_issues"] + df["closed_issues_count"]
    df["issue_resolution_rate"] = np.where(
        total_issues > 0,
        df["closed_issues_count"] / total_issues,
        0.0,
    ).round(4)

    # Stars per day of life — normalized popularity growth
    df["stars_per_day"] = np.where(
        df["repo_age_days"] > 0,
        df["stars"] / df["repo_age_days"],
        0.0,
    ).round(6)

    # Forks-to-stars ratio — adoption vs. interest ratio
    df["fork_star_ratio"] = np.where(
        df["stars"] > 0,
        df["forks"] / df["stars"],
        0.0,
    ).round(4)

    # Activity score — composite of recent signals (0–1 normalized)
    df["activity_score"] = (
        df["weekly_commit_avg"].clip(upper=20) / 20 * 0.35
        + (1 - df["days_since_last_push"].clip(upper=730) / 730) * 0.35
        + df["issue_resolution_rate"] * 0.15
        + df["has_ci"].astype(float) * 0.15
    ).round(4)

    # Topic count — breadth of self-described ecosystem
    df["topic_count"] = df["topics"].apply(
        lambda x: len(str(x).split("|")) if str(x).strip() else 0
    )

    # Has documentation flag
    df["has_readme"] = (df["readme_length"] > 200).astype(int)

    return df


# ---------------------------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------------------------

def split_dataset(
    labeled_path: str = LABELED_PATH,
    splits_dir: str = SPLITS_DIR,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the labeled dataset into train, validation, and test sets.

    Uses stratified splitting to preserve label distribution across splits.
    The test set is held out and never used during training.
    """

    df = pd.read_csv(labeled_path)
    df = df.dropna(subset=["label", "summary"])
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Total labeled samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state,
    )

    # Second split: val vs test
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    # Save splits
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(f"{splits_dir}/train.csv", index=False)
    val_df.to_csv(f"{splits_dir}/val.csv", index=False)
    test_df.to_csv(f"{splits_dir}/test.csv", index=False)

    print(f"Split complete (stratified):")
    print(f"  Train : {len(train_df)} ({len(train_df)/len(df):.0%})")
    print(f"  Val   : {len(val_df)} ({len(val_df)/len(df):.0%})")
    print(f"  Test  : {len(test_df)} ({len(test_df)/len(df):.0%})")
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
    print(f"\nVal label distribution:\n{val_df['label'].value_counts()}")
    print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
    print(f"\nSaved to: {splits_dir}/")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(raw_path: str = RAW_PATH, output_path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nPreprocessing complete.")
    print(f"  Shape: {df.shape}")
    print(f"  Saved to: {output_path}")

    return df


if __name__ == "__main__":
    run()
    split_dataset()
