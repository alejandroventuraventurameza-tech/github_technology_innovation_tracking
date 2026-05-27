# GitHub Technology Innovation & Ecosystem Tracking

**Track B — Technology Innovation & Ecosystem Tracking**  
Course: Machine Learning / NLP / Applied AI

---

## What does this project do?

TODO: Brief description of the system.

---

## Track Selected

**Track B — Technology Innovation & Ecosystem Tracking**

This system analyzes GitHub repository signals to identify and classify technologies into four categories:

- **Emerging** — technologies with accelerating growth and community momentum
- **Mature** — stable, widely adopted ecosystems
- **Declining** — technologies with decreasing activity and interest
- **Experimental / Niche** — early-stage or domain-specific tools with limited but focused adoption

---

## What repositories were analyzed?

TODO: Describe the repository selection strategy and domains covered.

---

## Which GitHub signals were used?

TODO: List and justify the 6+ signals extracted per repository.

---

## How were repository summaries created?

TODO: Explain the textual representation strategy used for LLM and BERT input.

---

## How were prompts designed?

TODO: Explain prompt design for weak labeling with DeepSeek.

---

## How was the dataset split?

- Train: 70%
- Validation: 15%
- Test: 15%

---

## Which BERT model was used?

TODO: e.g., DistilBERT, MiniLM, DeBERTa-v3-small.

---

## Final Metrics

TODO: Fill after model training.

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Precision | —     |
| Recall    | —     |
| F1-score  | —     |

---

## Main Limitations

TODO: Discuss limitations of weak supervision, data collection biases, etc.

---

## Possible Business Applications

TODO: Investors, consulting firms, governments, technology researchers.

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

# 4. Run the pipeline
# TODO: add pipeline steps
```

---

## How to run the Streamlit app

```bash
streamlit run app.py
```

---
