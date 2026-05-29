"""
summarization.py
----------------
Converts cleaned repository metadata into natural language summaries
usable as input for:
  - LLM weak labeling (Stage 3)
  - BERT fine-tuning (Stage 5)

Each summary is a short paragraph describing the repository's signals
in plain English, making it easy for the LLM to reason about its
technological maturity and momentum.

Track B — Technology Innovation & Ecosystem Tracking
"""

import os
import pandas as pd

PROCESSED_PATH = "data/processed/repositories_clean.csv"
SUMMARIZED_PATH = "data/processed/repositories_summarized.csv"


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(row: pd.Series) -> str:
    """
    Convert a repository row into a plain-English paragraph.

    Design rationale:
    - Natural language is richer input for LLMs than raw CSV values.
    - Each signal is verbalized so the LLM can apply commonsense reasoning
      (e.g., "last updated 3 years ago" clearly signals declining activity).
    - The summary is also used as BERT input, so it must stay concise
      (under ~300 tokens) while covering the most informative signals.
    """

    name = row["full_name"]
    desc = row["description"] if row["description"] else "no description provided"
    language = row["language"]
    topics_raw = str(row["topics"]) if pd.notna(row["topics"]) else ""
    topics = topics_raw.replace("|", ", ") if topics_raw.strip() else "none"
    license_ = row["license"]

    stars = int(row["stars"])
    forks = int(row["forks"])
    contributors = int(row["contributors_count"])
    weekly_commits = round(float(row["weekly_commit_avg"]), 1)
    releases = int(row["releases_count"])
    open_issues = int(row["open_issues"])
    closed_issues = int(row["closed_issues_count"])
    age_days = int(row["repo_age_days"])
    days_inactive = int(row["days_since_last_push"])
    has_ci = bool(row["has_ci"])
    readme_len = int(row["readme_length"])
    activity_score = round(float(row["activity_score"]), 2)

    # Verbalize age
    age_years = age_days / 365
    if age_years < 1:
        age_str = f"{age_days} days old"
    else:
        age_str = f"{age_years:.1f} years old"

    # Verbalize inactivity
    if days_inactive < 30:
        activity_str = "updated within the last month"
    elif days_inactive < 180:
        activity_str = f"last updated {days_inactive} days ago"
    elif days_inactive < 730:
        activity_str = f"last updated {days_inactive // 30} months ago"
    else:
        activity_str = f"last updated over {days_inactive // 365} year(s) ago"

    # Verbalize commit velocity
    if weekly_commits == 0:
        commit_str = "no recent commits"
    elif weekly_commits < 1:
        commit_str = "less than one commit per week on average"
    elif weekly_commits < 5:
        commit_str = f"{weekly_commits} commits per week on average"
    else:
        commit_str = f"very active with {weekly_commits} commits per week on average"

    # Verbalize documentation
    if readme_len == 0:
        doc_str = "no README documentation"
    elif readme_len < 500:
        doc_str = "minimal README documentation"
    elif readme_len < 3000:
        doc_str = "moderate README documentation"
    else:
        doc_str = "extensive README documentation"

    ci_str = "has CI/CD workflows configured" if has_ci else "no CI/CD workflows"

    summary = (
        f"Repository '{name}' is {age_str} and written primarily in {language}. "
        f"Description: {desc}. "
        f"Topics: {topics}. "
        f"It has {stars:,} stars and {forks:,} forks, with {contributors} contributor(s). "
        f"Development velocity: {commit_str}, and it was {activity_str}. "
        f"It has {releases} release(s), {open_issues} open issue(s), and {closed_issues} closed issue(s). "
        f"The project has {doc_str} and {ci_str}. "
        f"License: {license_}. "
        f"Overall activity score: {activity_score}/1.0."
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    processed_path: str = PROCESSED_PATH,
    output_path: str = SUMMARIZED_PATH,
) -> pd.DataFrame:

    df = pd.read_csv(processed_path)
    print(f"Building summaries for {len(df)} repositories...")

    df["summary"] = df.apply(build_summary, axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Summaries complete. Saved to: {output_path}")
    print(f"\nExample summary:\n")
    print(df["summary"].iloc[0])

    return df


if __name__ == "__main__":
    run()
