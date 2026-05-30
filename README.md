# GitHub Technology Innovation & Ecosystem Tracking

**Track B — Technology Innovation & Ecosystem Tracking**  
Course: Machine Learning / NLP / Applied AI

---

## What does this project do?

This system analyzes GitHub repositories from the **Econometrics & Macro Tooling** ecosystem and classifies them by their technological maturity stage using a **weak supervision NLP pipeline**. It combines GitHub API signals, LLM-generated weak labels (DeepSeek), and a fine-tuned DistilBERT classifier to automatically identify whether a tool is emerging, mature, declining, or experimental.

---

## Track Selected

**Track B — Technology Innovation & Ecosystem Tracking**

Repositories are classified into four categories:

- **Emerging** — technologies gaining momentum with accelerating activity and growing community
- **Mature** — stable, widely adopted ecosystems with sustained development
- **Declining** — tools with decreasing activity, long inactivity periods, and shrinking engagement
- **Experimental / Niche** — early-stage or highly specialized tools with limited but focused adoption

---

## What repositories were analyzed?

587 GitHub repositories from the **Econometrics & Macro Tooling** ecosystem, collected via 15 keyword/topic queries including: `econometrics`, `macroeconomics`, `dsge`, `panel-data`, `time-series`, `cointegration`, `nowcasting`, `var-model`, `causal-inference`, `structural-estimation`.

**Selection criteria:** minimum 5 stars to filter noise. Repositories span Python and R tools used in quantitative economics, macroeconomic modeling, and causal inference research.

---

## Which GitHub signals were used?

12 repository-level signals were extracted:

| Signal | Description |
|--------|-------------|
| `stars` | Popularity proxy — community interest |
| `forks` | Adoption proxy — how many build on this |
| `contributors_count` | Ecosystem breadth |
| `weekly_commit_avg` | Development velocity (last 52 weeks) |
| `days_since_last_push` | Activity recency — proxy for abandonment |
| `repo_age_days` | Maturity proxy |
| `releases_count` | Production-readiness |
| `open_issues` | Community activity |
| `closed_issues_count` | Community responsiveness |
| `has_ci` | Engineering maturity (GitHub Actions) |
| `readme_length` | Documentation quality |
| `watchers` | Sustained interest |

6 derived features were also engineered: `activity_score`, `stars_per_day`, `fork_star_ratio`, `issue_resolution_rate`, `topic_count`, `has_readme`.

---

## How were repository summaries created?

Each repository's signals were verbalized into a plain-English paragraph. Example:

> *"Repository 'statsmodels/statsmodels' is 15.0 years old and written primarily in Python. It has 11,435 stars and 3,343 forks, with 457 contributors. Development velocity: very active with 6.2 commits per week on average, and it was updated within the last month. It has 31 releases, 2973 open issues, and 1 closed issue. The project has extensive README documentation and has CI/CD workflows configured. Overall activity score: 0.61/1.0."*

This representation was chosen because natural language summaries are richer input for LLMs than raw tabular data, and allow the model to apply commonsense reasoning about repository maturity.

---

## How were prompts designed?

Two prompt versions were tested as part of the methodological sensitivity analysis (Track B Q4):

- **Baseline prompt:** strict category definitions → severe class imbalance (declining: 337, emerging: 15)
- **Refined prompt:** more nuanced definitions with explicit balance guidance → improved distribution (declining: 258, experimental: 217, mature: 85, emerging: 27)

The refined prompt included explicit instructions to distribute labels thoughtfully and avoid classifying every inactive repo as declining.

---

## How was the dataset split?

Stratified split to preserve label distribution:

| Split | Size | % |
|-------|------|---|
| Train | 410 | 70% |
| Validation | 88 | 15% |
| Test | 89 | 15% |

---

## Which BERT model was used?

**DistilBERT** (`distilbert-base-uncased`) — a lightweight transformer that retains 97% of BERT's performance at 40% fewer parameters. Chosen for its efficiency on CPU training and suitability for short text classification tasks.

Class imbalance was handled via **weighted cross-entropy loss** (emerging: 2.576×, mature: 0.83×, declining: 0.272×, experimental: 0.322×).

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 55.06% |
| Macro F1 | 0.41 |
| F1 — declining | 0.64 |
| F1 — mature | 0.55 |
| F1 — experimental | 0.47 |
| F1 — emerging | 0.00 |

Best model saved at epoch 2 (Val F1 macro = 0.52).

---

## Main Limitations

- **Weak labels:** DeepSeek-generated labels may be noisy — the model learns from imperfect annotations
- **Class imbalance:** `emerging` class has only 27 samples, leading to F1=0.00 on the test set
- **CPU training:** limits batch size and number of epochs
- **Selection bias:** GitHub search favors popular repos — obscure tools may be underrepresented
- **Static snapshot:** data reflects a single collection point, not longitudinal trends

---

## Possible Business Applications

- **Central banks & governments:** monitor which econometric tools are gaining adoption to inform software procurement and research investment decisions
- **Academic institutions:** identify emerging methodological tools for curriculum updates
- **Consulting firms & think tanks:** track which quantitative methods are gaining or losing traction in the research community
- **Investors in developer tooling:** identify growing open-source ecosystems with potential for commercialization

---

## How to run the project

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/github_technology_innovation_tracking.git
cd github_technology_innovation_tracking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your GitHub token and DeepSeek API key

# 4. Run the full pipeline
python -m src.github_collector      # Stage 1: Data collection
python -m src.preprocessing         # Stage 2: Cleaning + feature engineering
python -m src.summarization         # Stage 2: Repository summarization
python -m src.llm_labeling          # Stage 3: Weak labeling with DeepSeek
python -c "from src.preprocessing import split_dataset; split_dataset()"  # Stage 4: Splits
python -m src.train                 # Stage 5: DistilBERT fine-tuning
python -c "from src.evaluation import run; run()"  # Stage 6: Evaluation
```

---

## How to run the Streamlit app

```bash
streamlit run app.py
```

---
