# Cybersecurity Threat Intelligence QA Dataset

A verified binary forecasting dataset covering cybersecurity threats, vulnerabilities, and incident response — generated using the [Lightning Rod Labs](https://www.lightningrod.ai/) SDK.

**455 verified QA pairs** across 14 cybersecurity subcategories with 0.97 mean confidence and near-perfect label balance.

## Overview

This project uses the Lightning Rod SDK's forecasting pipeline to generate high-quality, labeled binary questions about real-world cybersecurity events. Each entry includes a question, a verified yes/no label, detailed reasoning with source citations, and the original news article that seeded it.

The dataset covers 90 days of cybersecurity news (November 2025 – February 2026) and spans vulnerability disclosures, ransomware incidents, threat actor attribution, regulatory actions, data breaches, and supply chain attacks.

| Metric | Value |
|---|---|
| Total verified pairs | 455 |
| Label balance | 53% Yes / 47% No |
| Mean confidence | 0.97 |
| Topic coverage | 14/14 query categories |
| Avg reasoning length | ~1,350 characters |
| Answer type | Binary (Yes/No) forecasting |

## Pipeline

The generation pipeline follows the Lightning Rod SDK's core architecture:

```
NewsSeedGenerator → ForwardLookingQuestionGenerator → WebSearchLabeler
```

1. **NewsSeedGenerator** pulls recent cybersecurity news articles from GDELT's news index using 14 targeted search queries
2. **ForwardLookingQuestionGenerator** creates binary forecasting questions with near-term resolution dates (30–90 days)
3. **WebSearchLabeler** verifies each question against web sources, producing a label (0 or 1), confidence score, and detailed reasoning with citations

Questions with undetermined labels, date ordering issues, or low confidence (< 0.90) are filtered out. Duplicate questions are removed.

## Search Queries

The 14 queries were designed to match GDELT's news headline indexing style for broad cybersecurity topic coverage:

| Category | Queries |
|---|---|
| Vulnerability / Patch | `critical vulnerability patch update`, `CISA known exploited vulnerability`, `Microsoft security update zero-day` |
| Ransomware / Attacks | `ransomware attack hospital`, `ransomware attack government`, `cyberattack critical infrastructure` |
| Threat Actors | `Chinese hackers cyber espionage`, `Russian cyber attack` |
| Regulatory / Policy | `cybersecurity regulation law`, `SEC cyber breach disclosure` |
| Data Breaches | `data breach million records`, `healthcare data breach` |
| Supply Chain / Cloud | `software supply chain hack`, `cloud security breach` |

## Dataset Schema

| Column | Description |
|---|---|
| `sample_id` | Unique identifier (UUID) |
| `question` | Binary forecasting question |
| `date_close` | Resolution deadline |
| `event_date` | Date of the underlying event |
| `resolution_criteria` | Detailed criteria for yes/no resolution |
| `prediction_date` | When the question was generated |
| `label` | Verified answer: 1 (Yes) or 0 (No) |
| `label_confidence` | Confidence score (0.90–1.00) |
| `resolution_date` | When the label was determined |
| `label_reasoning` | Multi-paragraph reasoning with evidence |
| `answer_sources` | Source URLs used for verification |
| `seed_text` | Original news article text |
| `seed_url` | Source URL of the news article |
| `seed_creation_date` | Publication date of the source article |
| `search_query` | Which of the 14 queries found this article |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/BBALabs/cybersec-threat-intel-qa.git
cd cybersec-threat-intel-qa

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install lightningrod-ai pandas pyarrow

# Set your API key (get one at dashboard.lightningrod.ai)
export LIGHTNINGROD_API_KEY="your-key-here"

# Generate a test batch
python generate_cybersec_dataset.py --test

# Generate a full dataset
python generate_cybersec_dataset.py --full
```

## Usage

```bash
# Test run (10 questions, ~$0.45)
python generate_cybersec_dataset.py --test

# Small validation run (100 questions, ~$4.50)
python generate_cybersec_dataset.py --small

# Full production run (2000 questions, ~$90)
python generate_cybersec_dataset.py --full

# Custom size
python generate_cybersec_dataset.py --custom 500
```

Output files are saved to `./output/` in JSON, CSV, and Parquet formats.

## Results

From ~2,200 raw generated questions:
- **455 passed verification** (valid label, high confidence, no duplicates)
- **Label distribution:** 240 Yes (53%) / 215 No (47%)
- **All 14 query categories** produced valid results
- **Top categories by volume:** CISA KEV additions, Microsoft zero-days, cybersecurity regulation, vulnerability patches

## Dataset

The published dataset is available on HuggingFace: [blackboxanalytics/cybersec-threat-intel-qa](https://huggingface.co/datasets/blackboxanalytics/cybersec-threat-intel-qa)

## Built With

- [Lightning Rod Labs SDK](https://github.com/lightning-rod-labs/lightningrod-python-sdk) — forecasting dataset generation
- [Lightning Rod Labs](https://www.lightningrod.ai/) — platform and API
- Based on the research: [Future-as-Label: Scalable Supervision from Real-World Outcomes](https://arxiv.org/abs/2601.06336)

## Author

Tony Winslow — [Black Box Analytics](https://github.com/BBALabs)

## License

MIT
