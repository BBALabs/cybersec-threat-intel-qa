#!/usr/bin/env python3
"""
Lightning Rod Labs — Cybersecurity Threat Intelligence QA Dataset Generator
===========================================================================
Project:  Milestone 1 for Lightning Rod Labs (DevRel)
Author:   Tony Winslow | Black Box Analytics
Server:   Umbra (blackbox) — /home/umbra/lightningrod-cybersec/
Venv:     /home/umbra/lightningrod-cybersec/.venv/

Usage:
    source .venv/bin/activate && source .env
    python generate_cybersec_dataset.py --test       # 10 questions
    python generate_cybersec_dataset.py --small      # 100 questions
    python generate_cybersec_dataset.py --full       # 2000 questions

v3.1 — Production-ready, fully verified against SDK docs + contract.
    Changes from v3:
    - Updated question examples to 2026 dates (matches 90-day news window)
    - Added sample_id (UUID) column matching LR's published dataset format
    - Added duplicate question detection + removal
    - Enhanced completion summary for quality reporting
    Verified:
    - SDK import: BinaryAnswerType() confirmed against GitHub README + Umbra test
    - Output format: superset of LR's HuggingFace dataset columns (richer)
    - Cost: ~$90 for --full (2000 raw), within $100 budget
    - Contract: targets 500-1000 valid pairs, expect ~700 at 34.9% valid rate
"""

import os
import sys
import json
import uuid
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. API Key
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("LIGHTNINGROD_API_KEY")
if not API_KEY:
    print("[ERROR] LIGHTNINGROD_API_KEY not set. Run: source .env")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. SDK Imports — BinaryAnswerType confirmed working on Umbra (v0.1.x)
# ---------------------------------------------------------------------------
try:
    from lightningrod import (
        LightningRod,
        BinaryAnswerType,
        QuestionPipeline,
        NewsSeedGenerator,
        ForwardLookingQuestionGenerator,
        WebSearchLabeler,
    )
    ANSWER_TYPE = BinaryAnswerType()
    print("[OK] SDK loaded (BinaryAnswerType)")
except ImportError as e:
    print(f"[ERROR] Failed to import SDK: {e}")
    print("  Run: pip install lightningrod-ai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------

# v3 QUERIES: 14 queries using news-headline phrasing for GDELT coverage.
# v2 had 8 queries but only 3 hit. These match how journalists title articles.
CYBERSEC_QUERIES = [
    # Vulnerability / Patch management
    "critical vulnerability patch update",
    "CISA known exploited vulnerability",
    "Microsoft security update zero-day",

    # Ransomware / Attacks
    "ransomware attack hospital",
    "ransomware attack government",
    "cyberattack critical infrastructure",

    # Threat actors / Attribution
    "Chinese hackers cyber espionage",
    "Russian cyber attack",

    # Regulatory / Policy
    "cybersecurity regulation law",
    "SEC cyber breach disclosure",

    # Data breaches
    "data breach million records",
    "healthcare data breach",

    # Supply chain / Cloud
    "software supply chain hack",
    "cloud security breach",
]

QUESTION_INSTRUCTIONS = """Generate binary (yes/no) forecasting questions about cybersecurity 
events from the provided news articles.

CRITICAL RULES FOR RESOLUTION DATES:
- Set resolution dates within 30 to 90 days of the article date
- Prefer dates that have ALREADY PASSED so the web labeler can verify the outcome
- Good: "by February 2026", "by March 15, 2026", "within 60 days"
- Bad: "by 2027", "by December 2026", "within the next year"

QUESTION QUALITY GUIDELINES:
- Reference specific entities: CVE IDs, company names, agency names, threat actor names
- Cover diverse outcomes: patch releases, KEV catalog additions, breach notifications, 
  enforcement actions, threat actor indictments, policy deadlines, vendor advisories
- Each question must be self-contained and verifiable from public sources
- Include explicit calendar dates for resolution, not vague timeframes

IMPORTANT: Generate questions where the resolution date has likely already passed
relative to today's date, so the labeler can determine a definitive yes or no answer."""

# Examples updated to 2026 dates to match the 90-day news window (Nov 2025 - Feb 2026)
QUESTION_EXAMPLES = [
    "Will CISA add CVE-2025-21298 to its Known Exploited Vulnerabilities catalog by March 1, 2026?",
    "Will Microsoft release an out-of-band security patch for the Outlook zero-day by February 15, 2026?",
    "Will the FBI or DOJ announce arrests related to LockBit ransomware leadership by April 2026?",
    "Will the SEC file enforcement action against a public company for delayed breach notification by March 2026?",
    "Will Google Project Zero publish a technical advisory on a new actively-exploited Chrome zero-day by March 15, 2026?",
    "Will a US hospital system publicly confirm paying a ransomware demand exceeding $1 million by March 2026?",
    "Will the EU formally designate a Chinese state-sponsored threat actor under the Cyber Diplomacy Toolbox by April 2026?",
]

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------
def build_pipeline():
    lr = LightningRod(api_key=API_KEY)
    pipeline = QuestionPipeline(
        seed_generator=NewsSeedGenerator(
            start_date=START_DATE,
            end_date=END_DATE,
            search_query=CYBERSEC_QUERIES,
        ),
        question_generator=ForwardLookingQuestionGenerator(
            instructions=QUESTION_INSTRUCTIONS,
            examples=QUESTION_EXAMPLES,
        ),
        labeler=WebSearchLabeler(answer_type=ANSWER_TYPE),
    )
    return lr, pipeline


# ---------------------------------------------------------------------------
# 5. Generate + Filter + Export
# ---------------------------------------------------------------------------
def generate_dataset(max_questions: int):
    print(f"\n{'='*65}")
    print(f"  Lightning Rod — Cybersec Threat Intel QA Dataset v3.1")
    print(f"  Max Questions:  {max_questions}")
    print(f"  Date Range:     {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Search Queries: {len(CYBERSEC_QUERIES)}")
    print(f"  Answer Type:    Binary (near-term forecasting)")
    print(f"{'='*65}\n")

    lr, pipeline = build_pipeline()

    print(f"[1/5] Running pipeline ({max_questions} questions)...")
    print(f"       This may take several minutes for large runs.")
    print(f"       Est. cost: ~${max_questions * 0.045:.2f}")
    start_time = datetime.now()

    try:
        dataset = lr.transforms.run(pipeline, max_questions=max_questions)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        print(f"  - Check credits: dashboard.lightningrod.ai")
        print(f"  - Try smaller batch: --custom 50")
        sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"       Pipeline completed in {elapsed:.0f}s")

    # --- Flatten ---
    print("[2/5] Flattening...")
    flat = dataset.flattened()
    total_raw = len(flat)
    print(f"       Raw entries: {total_raw}")

    # --- DataFrame ---
    print("[3/5] Processing and filtering...")
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas required. Run: pip install pandas")
        sys.exit(1)

    if hasattr(flat, 'to_pandas'):
        df_raw = flat.to_pandas()
    elif isinstance(flat, list):
        df_raw = pd.DataFrame(flat)
    else:
        df_raw = pd.DataFrame(flat)

    # --- Filter valid ---
    if 'is_valid' in df_raw.columns:
        df_valid = df_raw[df_raw['is_valid'] == True].copy()
        df_invalid = df_raw[df_raw['is_valid'] == False].copy()
    else:
        df_valid = df_raw.copy()
        df_invalid = pd.DataFrame()

    valid_count = len(df_valid)
    invalid_count = len(df_invalid)
    valid_pct = (valid_count / total_raw * 100) if total_raw > 0 else 0

    print(f"       Total:   {total_raw}")
    print(f"       Valid:   {valid_count} ({valid_pct:.1f}%)")
    print(f"       Removed: {invalid_count}")

    if invalid_count > 0 and 'meta.filter_reason' in df_invalid.columns:
        reasons = df_invalid['meta.filter_reason'].value_counts()
        for reason, count in reasons.items():
            print(f"         - {reason}: {count}")

    # --- Rename columns ---
    column_map = {
        'question.question_text': 'question',
        'question.date_close': 'date_close',
        'question.event_date': 'event_date',
        'question.resolution_criteria': 'resolution_criteria',
        'question.prediction_date': 'prediction_date',
        'label.label': 'label',
        'label.label_confidence': 'label_confidence',
        'label.resolution_date': 'resolution_date',
        'label.reasoning': 'label_reasoning',
        'label.answer_sources': 'answer_sources',
        'seed.seed_text': 'seed_text',
        'seed.url': 'seed_url',
        'seed.seed_creation_date': 'seed_creation_date',
        'seed.search_query': 'search_query',
    }

    keep_cols = [c for c in column_map.keys() if c in df_valid.columns]
    df_clean = df_valid[keep_cols].rename(columns=column_map)

    # --- Add sample_id (matches LR's published dataset convention) ---
    df_clean.insert(0, 'sample_id', [str(uuid.uuid4()) for _ in range(len(df_clean))])

    # --- Deduplicate by question text ---
    pre_dedup = len(df_clean)
    if 'question' in df_clean.columns and len(df_clean) > 0:
        df_clean = df_clean.drop_duplicates(subset='question', keep='first').reset_index(drop=True)
    post_dedup = len(df_clean)
    if pre_dedup != post_dedup:
        print(f"       Deduped:  removed {pre_dedup - post_dedup} duplicate questions")

    # Raw with renamed cols (for debug/analysis)
    keep_cols_raw = [c for c in column_map.keys() if c in df_raw.columns]
    extra_raw = ['is_valid', 'meta.filter_reason', 'meta.processing_time_ms']
    keep_cols_raw += [c for c in extra_raw if c in df_raw.columns]
    df_raw_renamed = df_raw[keep_cols_raw].rename(
        columns={**column_map, 'meta.filter_reason': 'filter_reason',
                 'meta.processing_time_ms': 'processing_time_ms'}
    )

    # --- Stats ---
    print(f"\n[4/5] Dataset Statistics:")
    print(f"       Final rows:  {len(df_clean)}")
    print(f"       Columns:     {list(df_clean.columns)}")

    if 'label' in df_clean.columns and len(df_clean) > 0:
        label_dist = df_clean['label'].value_counts()
        print(f"       Labels:")
        for label, count in label_dist.items():
            pct = count / len(df_clean) * 100
            label_name = {0: "No", 1: "Yes", "0": "No", "1": "Yes"}.get(label, label)
            print(f"         {label} ({label_name}): {count} ({pct:.1f}%)")

    if 'label_confidence' in df_clean.columns and len(df_clean) > 0:
        conf = pd.to_numeric(df_clean['label_confidence'], errors='coerce')
        print(f"       Confidence: mean={conf.mean():.2f}, min={conf.min():.2f}, max={conf.max():.2f}")

    if 'search_query' in df_clean.columns and len(df_clean) > 0:
        query_dist = df_clean['search_query'].value_counts()
        print(f"       Query coverage ({len(query_dist)}/{len(CYBERSEC_QUERIES)} queries hit):")
        for query, count in query_dist.items():
            print(f"         [{count:3d}] {query}")

    if 'question' in df_clean.columns and len(df_clean) > 0:
        q_lens = df_clean['question'].str.len()
        print(f"       Question length: mean={q_lens.mean():.0f}, min={q_lens.min()}, max={q_lens.max()} chars")

    if 'label_reasoning' in df_clean.columns and len(df_clean) > 0:
        r_lens = df_clean['label_reasoning'].str.len()
        print(f"       Reasoning length: mean={r_lens.mean():.0f}, min={r_lens.min()}, max={r_lens.max()} chars")

    # --- Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[5/5] Exporting...")

    # Clean JSON (valid only, HuggingFace-ready)
    json_path = OUTPUT_DIR / f"cybersec_threat_intel_qa_{timestamp}.json"
    df_clean.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(f"       JSON:      {json_path} ({len(df_clean)} rows)")

    # Clean CSV (valid only)
    csv_path = OUTPUT_DIR / f"cybersec_threat_intel_qa_{timestamp}.csv"
    df_clean.to_csv(csv_path, index=False)
    print(f"       CSV:       {csv_path}")

    # Parquet (for HuggingFace upload)
    try:
        parquet_path = OUTPUT_DIR / f"cybersec_threat_intel_qa_{timestamp}.parquet"
        df_clean.to_parquet(parquet_path, index=False)
        print(f"       Parquet:   {parquet_path}")
    except Exception as e:
        print(f"       [WARN] Parquet failed ({e}). Run: pip install pyarrow")

    # Raw CSV (all entries including invalid, for analysis)
    raw_path = OUTPUT_DIR / f"cybersec_threat_intel_qa_{timestamp}_ALL.csv"
    df_raw_renamed.to_csv(raw_path, index=False)
    print(f"       Raw (all): {raw_path} ({len(df_raw_renamed)} rows)")

    # --- Completion Summary ---
    if len(df_clean) > 0:
        cost_est = total_raw * 0.045
        print(f"\n{'='*65}")
        print(f"  COMPLETE — {len(df_clean)} valid QA pairs")
        print(f"  Valid rate: {valid_pct:.1f}% ({valid_count}/{total_raw})")
        print(f"  Runtime: {elapsed:.0f}s ({elapsed/total_raw:.1f}s per question)")
        print(f"  Est. cost: ~${cost_est:.2f}")
        if 'label' in df_clean.columns:
            yes_ct = ((df_clean['label'] == 1) | (df_clean['label'] == '1')).sum()
            no_ct = ((df_clean['label'] == 0) | (df_clean['label'] == '0')).sum()
            print(f"  Label balance: {yes_ct} Yes / {no_ct} No ({yes_ct/(yes_ct+no_ct)*100:.0f}%/{no_ct/(yes_ct+no_ct)*100:.0f}%)")
        if 'search_query' in df_clean.columns:
            print(f"  Topic coverage: {df_clean['search_query'].nunique()}/{len(CYBERSEC_QUERIES)} queries")
        print(f"{'='*65}")

        print(f"\n--- Sample Valid Entries ---\n")
        for i, (_, row) in enumerate(df_clean.head(5).iterrows()):
            print(f"[{i+1}] {'-'*55}")
            print(f"  ID:    {row.get('sample_id', '?')}")
            print(f"  Q:     {str(row.get('question', ''))[:250]}")
            print(f"  Label: {row.get('label', '?')} (conf: {row.get('label_confidence', '?')})")
            reason = str(row.get('label_reasoning', ''))
            print(f"  Why:   {reason[:350]}{'...' if len(reason) > 350 else ''}")
            print(f"  Seed:  {row.get('search_query', '?')}")
            print()
    else:
        print(f"\n{'='*65}")
        print(f"  WARNING — 0 valid entries from {total_raw} generated")
        print(f"  Check raw output: {raw_path}")
        print(f"  Common causes:")
        print(f"    - All questions had Undetermined labels (resolution dates too far out)")
        print(f"    - No GDELT news articles matched the search queries")
        print(f"    - API credits exhausted mid-run")
        print(f"{'='*65}")

    return df_clean


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Cybersecurity QA Dataset with Lightning Rod SDK (v3.1)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true",
                       help="Test: 10 questions (~$0.45)")
    group.add_argument("--small", action="store_true",
                       help="Small: 100 questions (~$4.50, ~35 valid)")
    group.add_argument("--full", action="store_true",
                       help="Full: 2000 questions (~$90, ~700 valid)")
    group.add_argument("--custom", type=int, metavar="N",
                       help="Custom: N questions")

    args = parser.parse_args()

    if args.test:
        max_q = 10
    elif args.small:
        max_q = 100
    elif args.full:
        max_q = 2000
    else:
        max_q = args.custom

    generate_dataset(max_questions=max_q)


if __name__ == "__main__":
    main()
