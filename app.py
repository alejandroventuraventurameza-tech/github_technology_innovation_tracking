"""
app.py
------
Streamlit application for GitHub Technology Innovation & Ecosystem Tracking.
Contains exactly 4 tabs as required by the assignment.

Track B — Technology Innovation & Ecosystem Tracking
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.utils import LABEL2ID, ID2LABEL
from src.visualization import (
    plot_label_distribution,
    plot_signal_by_label,
    plot_activity_score_distribution,
    plot_stars_vs_age,
    plot_confusion_matrix_fig,
    plot_per_class_f1,
    plot_baseline_vs_alternative,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GitHub Tech Ecosystem Tracker",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_labeled():
    return pd.read_csv("data/labeled/repositories_labeled.csv")


@st.cache_data
def load_test():
    return pd.read_csv("data/splits/test.csv")


@st.cache_data
def load_metrics():
    with open("output/metrics/test_metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/trained_models")
    model = DistilBertForSequenceClassification.from_pretrained("models/trained_models")
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_label(summary: str, tokenizer, model) -> tuple[str, dict]:
    inputs = tokenizer(
        summary, return_tensors="pt", truncation=True,
        max_length=256, padding="max_length"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_id = int(np.argmax(probs))
    label = ID2LABEL[pred_id]
    prob_dict = {ID2LABEL[i]: round(float(probs[i]), 3) for i in range(len(ID2LABEL))}
    return label, prob_dict


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("🔬 GitHub Technology Innovation & Ecosystem Tracking")
st.caption("Track B — Weak Supervision NLP Pipeline | Econometrics & Macro Tooling Ecosystem")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Problem & Methodology",
    "📊 Exploratory Analysis",
    "🤖 Model Results",
    "🔍 Interactive Exploration",
])


# ============================================================
# TAB 1 — Problem & Methodology
# ============================================================

with tab1:
    st.header("Problem & Methodology")

    st.subheader("Project Objective")
    st.markdown("""
    This system analyzes GitHub repositories from the **Econometrics & Macro Tooling** ecosystem
    and classifies them by their technological maturity stage using a **weak supervision NLP pipeline**.

    The goal is to help investors, researchers, governments, and consulting firms understand
    which tools in the quantitative economics space are **emerging**, **mature**, **declining**, or **experimental**.
    """)

    st.subheader("Repository Selection Methodology")
    st.markdown("""
    Repositories were collected via the **GitHub REST API** using 15 keyword/topic queries targeting:
    `econometrics`, `macroeconomics`, `dsge`, `panel-data`, `time-series`, `cointegration`,
    `nowcasting`, `var-model`, `causal-inference`, `structural-estimation`.

    **Selection criteria:** minimum 5 stars to filter noise. Final dataset: **587 repositories**.

    **Potential bias:** well-maintained repos are more discoverable via GitHub search.
    Abandoned or obscure repos may be underrepresented.
    """)

    st.subheader("GitHub Signals Used (12 features)")
    signals = {
        "stars": "Popularity proxy — measures community interest",
        "forks": "Adoption proxy — how many people build on this",
        "contributors_count": "Ecosystem breadth — number of unique contributors",
        "weekly_commit_avg": "Development velocity — average commits per week (last 52 weeks)",
        "days_since_last_push": "Activity recency — proxy for abandonment",
        "repo_age_days": "Maturity proxy — how long the project has existed",
        "releases_count": "Production-readiness — number of versioned releases",
        "open_issues": "Community activity — pending user engagement",
        "closed_issues_count": "Community responsiveness — issue resolution history",
        "has_ci": "Engineering maturity — presence of GitHub Actions workflows",
        "readme_length": "Documentation quality — length of README in characters",
        "watchers": "Sustained interest — users following the repo",
    }
    for signal, desc in signals.items():
        st.markdown(f"- **`{signal}`** — {desc}")

    st.subheader("Prompt Strategy")
    st.markdown("""
    Two prompt versions were tested (baseline vs. refined) as part of the **methodological sensitivity analysis** (Track B Q4):
    - **Baseline:** strict definitions, led to severe class imbalance (57% declining)
    - **Refined:** more nuanced category descriptions with explicit balance guidance, improved emerging class from 15 → 27 repos
    """)

    st.subheader("Dataset Construction")
    st.markdown("""
    1. GitHub API → 587 repos → `data/raw/repositories.csv`
    2. Preprocessing + feature engineering → `data/processed/repositories_clean.csv`
    3. Textual summarization → `data/processed/repositories_summarized.csv`
    4. DeepSeek LLM weak labeling → `data/labeled/repositories_labeled.csv`
    5. Stratified split 70/15/15 → `data/splits/`
    """)

    st.subheader("Limitations")
    st.markdown("""
    - **Weak labels:** DeepSeek labels may be noisy — the model learns from imperfect annotations
    - **Class imbalance:** `emerging` class has only 27 samples, limiting model performance on that class
    - **CPU training:** DistilBERT trained on CPU limits epochs and batch size
    - **Selection bias:** GitHub search favors popular repos — niche tools may be underrepresented
    - **Static snapshot:** data reflects a single point in time, not longitudinal trends
    """)


# ============================================================
# TAB 2 — Exploratory Analysis
# ============================================================

with tab2:
    st.header("Exploratory Analysis")

    try:
        df = load_labeled()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Repositories", len(df))
        col2.metric("Avg Stars", f"{df['stars'].mean():.0f}")
        col3.metric("Avg Contributors", f"{df['contributors_count'].mean():.1f}")
        col4.metric("Avg Weekly Commits", f"{df['weekly_commit_avg'].mean():.1f}")

        st.subheader("Label Distribution")
        st.markdown("""
        The distribution reflects the nature of the econometrics ecosystem: most tools are old and
        under-maintained (declining), while a smaller share represents active or niche projects.
        """)
        fig1 = plot_label_distribution(df)
        st.pyplot(fig1)

        st.subheader("Stars vs. Repository Age")
        st.markdown("""
        Emerging repos tend to accumulate stars quickly relative to their age.
        Mature repos are old with many stars. Declining repos are old with stagnant growth.
        """)
        fig2 = plot_stars_vs_age(df)
        st.pyplot(fig2)

        st.subheader("Activity Score by Label")
        st.markdown("""
        The composite activity score (commits, recency, CI, issue resolution) clearly
        separates emerging/mature repos from declining ones.
        """)
        fig3 = plot_signal_by_label(df, "activity_score", "Activity Score by Label")
        st.pyplot(fig3)

        st.subheader("Activity Score Distribution (KDE)")
        fig4 = plot_activity_score_distribution(df)
        st.pyplot(fig4)

        st.subheader("Signal Comparison")
        signal = st.selectbox("Select signal to compare across labels:", [
            "stars", "forks", "contributors_count", "weekly_commit_avg",
            "days_since_last_push", "releases_count", "readme_length",
        ])
        fig5 = plot_signal_by_label(df, signal, f"{signal} by Label")
        st.pyplot(fig5)

    except FileNotFoundError:
        st.warning("Labeled data not found. Run the full pipeline first.")


# ============================================================
# TAB 3 — Model Results
# ============================================================

with tab3:
    st.header("Model Results")

    try:
        metrics = load_metrics()
        test_df = load_test()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        col3.metric("Test Samples", len(test_df))

        st.subheader("Per-Class Performance")
        perf_df = pd.DataFrame(metrics["per_class"]).T
        st.dataframe(perf_df.style.format("{:.3f}", subset=["precision", "recall", "f1"]))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Confusion Matrix")
            st.image("output/figures/confusion_matrix.png", use_container_width=True)

        with col_b:
            st.subheader("F1-Score per Class")
            fig_f1 = plot_per_class_f1(metrics)
            st.pyplot(fig_f1)

        st.subheader("Baseline vs. Refined Prompt (Q4 — Methodological Sensitivity)")
        st.markdown("""
        We ran two labeling rounds with different prompt strategies. The refined prompt
        produced a more balanced distribution, particularly improving the `emerging` and
        `experimental` classes at the cost of some `declining` labels.
        """)
        fig_baseline = plot_baseline_vs_alternative()
        st.pyplot(fig_baseline)

        st.subheader("Error Analysis")
        st.markdown("""
        **Main failure mode:** the model struggles with `emerging` (F1=0.00) due to only
        4 test samples. `declining` and `experimental` are frequently confused because
        both share low-activity signals but differ in intentionality.
        """)

    except FileNotFoundError:
        st.warning("Metrics or test data not found. Run the evaluation pipeline first.")


# ============================================================
# TAB 4 — Interactive Repository Exploration
# ============================================================

with tab4:
    st.header("Interactive Repository Exploration")

    try:
        df = load_labeled()

        st.subheader("Filter & Search Repositories")
        col1, col2 = st.columns(2)
        with col1:
            selected_labels = st.multiselect(
                "Filter by label:",
                options=["emerging", "mature", "declining", "experimental"],
                default=["emerging", "mature", "declining", "experimental"],
            )
        with col2:
            search_query = st.text_input("Search by name or description:", "")

        filtered = df[df["label"].isin(selected_labels)]
        if search_query:
            mask = (
                filtered["full_name"].str.contains(search_query, case=False, na=False) |
                filtered["description"].str.contains(search_query, case=False, na=False)
            )
            filtered = filtered[mask]

        st.markdown(f"**{len(filtered)} repositories found**")
        display_cols = ["full_name", "label", "stars", "forks", "contributors_count",
                        "weekly_commit_avg", "days_since_last_push", "activity_score"]
        st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

        st.divider()
        st.subheader("Live Model Prediction")
        st.markdown("Enter a repository summary and the model will classify it in real time.")

        example_summary = st.text_area(
            "Repository summary:",
            value="Repository 'example/econometrics-tool' is 2.5 years old and written primarily in Python. "
                  "It has 450 stars and 89 forks, with 12 contributors. Development velocity: 3.2 commits "
                  "per week on average, and it was updated within the last month. It has 8 releases, "
                  "45 open issues, and 120 closed issues. The project has extensive README documentation "
                  "and has CI/CD workflows configured. License: MIT. Overall activity score: 0.72/1.0.",
            height=150,
        )

        if st.button("Predict", type="primary"):
            with st.spinner("Running model..."):
                tokenizer, model = load_model()
                label, probs = predict_label(example_summary, tokenizer, model)

            st.success(f"Predicted label: **{label.upper()}**")
            prob_df = pd.DataFrame(list(probs.items()), columns=["Label", "Probability"])
            prob_df = prob_df.sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Label"))

    except FileNotFoundError:
        st.warning("Labeled data not found. Run the full pipeline first.")
