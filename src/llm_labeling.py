"""
llm_labeling.py
---------------
Weak labeling stage: uses DeepSeek API to classify each repository
into one of four technology maturity categories.

Categories (Track B — Technology Innovation & Ecosystem Tracking):
  - emerging      : fast-growing, recent momentum, high activity
  - mature        : stable, widely adopted, sustained development
  - declining     : decreasing activity, low recent commits, aging
  - experimental  : niche, early-stage, low adoption but focused

The LLM acts as the initial annotator. Labels are "weak" because they
are machine-generated and may contain noise — this is intentional in
weak supervision pipelines, where the BERT model learns to generalize
beyond individual label errors.

Track B — Technology Innovation & Ecosystem Tracking
"""

import os
import re
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# DeepSeek uses the OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SUMMARIZED_PATH = "data/processed/repositories_summarized.csv"
LABELED_PATH = "data/labeled/repositories_labeled.csv"
LABELED_BASELINE_PATH = "data/labeled/repositories_labeled_baseline.csv"

VALID_LABELS = {"emerging", "mature", "declining", "experimental"}

# ---------------------------------------------------------------------------
# Prompt design
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert technology analyst specializing in open-source ecosystem intelligence.
Your task is to classify GitHub repositories by their technology maturity stage based on repository signals.

You must assign exactly one of these four labels. Apply them with a BALANCED perspective — not every
inactive repo is declining, and not every old repo is mature. Read all signals carefully.

- emerging: The technology is gaining momentum and shows growth signals. This includes:
  repos created 1-6 years ago with consistent or increasing activity, growing star/fork counts,
  active commit history (at least 1 commit/week on average), a growing contributor base, and
  recent releases. Even repos with moderate star counts (100-2000) can be emerging if their
  activity trend is upward and their topic is gaining relevance in the ecosystem.

- mature: The technology is stable, widely adopted, and well-established. Signs include:
  long history (5+ years), large star/fork counts (2000+ stars), many contributors (20+),
  consistent releases, strong documentation, and sustained but not explosive development.
  The project is a reference in its domain.

- declining: The technology is clearly losing momentum. Signs include:
  very low or zero recent commits, inactivity for more than 18 months, few or no recent
  releases, stagnant or shrinking community engagement, and outdated dependencies or documentation.
  Reserve this label for repos that show clear abandonment signals.

- experimental: The technology is niche, highly specialized, or early-stage with limited
  adoption. Signs include: low star counts (under 150), very small contributor base (1-3 people),
  highly domain-specific topics, few or no releases, but may show focused and intentional activity.
  This is for repos that are functional but have not yet gained broader ecosystem traction.

Important: Distribute labels thoughtfully. Not everything in an old field is declining —
many tools are mature or emerging within their niche.

Respond with ONLY the label — one word, lowercase. No explanation."""

USER_PROMPT_TEMPLATE = """Classify this repository:

{summary}

Label:"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def classify_repository(summary: str, retries: int = 3) -> str:
    """Send a summary to DeepSeek and return the predicted label."""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(summary=summary)},
                ],
                max_tokens=10,
                temperature=0.0,  # deterministic — important for reproducibility
            )

            raw = response.choices[0].message.content.strip().lower()

            # Clean any punctuation the model may add
            label = re.sub(r"[^a-z]", "", raw)

            if label in VALID_LABELS:
                return label
            else:
                print(f"  Unexpected label '{raw}', retrying...")
                time.sleep(1)

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return "experimental"  # fallback label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    summarized_path: str = SUMMARIZED_PATH,
    output_path: str = LABELED_PATH,
    sample: int = None,
) -> pd.DataFrame:
    """
    Label all repositories using DeepSeek weak supervision.

    Args:
        sample: if set, only label a random sample (useful for testing).
    """

    df = pd.read_csv(summarized_path)

    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Sampling {sample} repositories for labeling.")

    print(f"Labeling {len(df)} repositories with DeepSeek...")
    print(f"Estimated cost: ~${len(df) * 150 * 0.00000027:.4f} USD\n")

    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = classify_repository(row["summary"])
        labels.append(label)
        time.sleep(0.3)  # avoid rate limits

    df["label"] = labels

    # Summary statistics
    print(f"\nLabeling complete.")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df


if __name__ == "__main__":
    # To test with a small sample first, use: run(sample=10)
    # For full labeling, use: run()
    run()
