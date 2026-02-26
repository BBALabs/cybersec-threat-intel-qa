#!/usr/bin/env python3
"""
Cybersecurity Threat Intelligence QA — Fine-Tuning Evaluation
=============================================================
Project:  Milestone 1 for Lightning Rod Labs (DevRel)
Author:   Tony Winslow | Black Box Analytics

Evaluates the cybersec-threat-intel-qa dataset by comparing:
  - Zero-shot: Model answers cold with no examples
  - Few-shot (5 examples): Model gets relevant examples from the dataset first

Uses Ollama API (local) — no additional GPU resources needed.

Usage:
    python eval_cybersec_dataset.py
    python eval_cybersec_dataset.py --test-size 20    # smaller test run
    python eval_cybersec_dataset.py --model qwen2.5:32b-instruct-q4_K_M
"""

import json
import random
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:32b-instruct-q4_K_M"
DATASET_PATH = Path("/home/umbra/lightningrod-cybersec/output/cybersec_threat_intel_final.json")
OUTPUT_DIR = Path("/home/umbra/lightningrod-cybersec/output")
NUM_FEW_SHOT = 5
SEED = 42


# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------
def query_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    """Send prompt to Ollama and return response text."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
def zero_shot_prompt(question: str) -> str:
    return f"""You are a cybersecurity threat intelligence analyst. Answer the following binary forecasting question with ONLY "Yes" or "No" on the first line, followed by a brief explanation.

Question: {question}

Answer:"""


def few_shot_prompt(question: str, examples: list) -> str:
    example_block = ""
    for i, ex in enumerate(examples, 1):
        label_word = "Yes" if str(ex["label"]) == "1" else "No"
        reasoning = str(ex.get("label_reasoning", ""))[:300]
        example_block += f"""Example {i}:
Question: {ex['question']}
Answer: {label_word}
Reasoning: {reasoning}

"""

    return f"""You are a cybersecurity threat intelligence analyst. Below are {len(examples)} examples of binary forecasting questions with verified answers. Use these as reference for the format and reasoning style.

{example_block}Now answer the following question with ONLY "Yes" or "No" on the first line, followed by a brief explanation.

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Parse model response to Yes/No
# ---------------------------------------------------------------------------
def parse_prediction(response: str) -> str:
    """Extract Yes/No from model response. Returns 'Yes', 'No', or 'Unknown'."""
    if not response or response.startswith("[ERROR]"):
        return "Unknown"

    # Check first line
    first_line = response.strip().split("\n")[0].strip().lower()
    # Remove common prefixes
    for prefix in ["answer:", "**answer:**", "**"]:
        first_line = first_line.replace(prefix, "").strip()

    if first_line.startswith("yes"):
        return "Yes"
    elif first_line.startswith("no"):
        return "No"

    # Fallback: check entire response
    response_lower = response.lower()
    yes_count = response_lower.count(" yes") + response_lower.count("yes,") + response_lower.count("yes.")
    no_count = response_lower.count(" no") + response_lower.count("no,") + response_lower.count("no.")

    if yes_count > no_count:
        return "Yes"
    elif no_count > yes_count:
        return "No"

    return "Unknown"


# ---------------------------------------------------------------------------
# Select few-shot examples (diverse topics)
# ---------------------------------------------------------------------------
def select_few_shot_examples(train_pool: list, test_entry: dict, n: int = 5) -> list:
    """Pick n examples from train pool, preferring diverse search queries."""
    test_query = test_entry.get("search_query", "")

    # Get examples from different queries than the test question
    other_query = [e for e in train_pool if e.get("search_query", "") != test_query]
    same_query = [e for e in train_pool if e.get("search_query", "") == test_query]

    # Mix: 3-4 from other topics + 1-2 from same topic (if available)
    selected = []
    if other_query:
        # Try to get diverse queries
        seen_queries = set()
        for e in random.sample(other_query, min(len(other_query), n * 3)):
            q = e.get("search_query", "")
            if q not in seen_queries and len(selected) < min(n - 1, 4):
                selected.append(e)
                seen_queries.add(q)

    # Fill remainder from same query
    remaining = n - len(selected)
    if remaining > 0 and same_query:
        selected.extend(random.sample(same_query, min(len(same_query), remaining)))

    # If still short, fill from any
    remaining = n - len(selected)
    if remaining > 0:
        available = [e for e in train_pool if e not in selected]
        selected.extend(random.sample(available, min(len(available), remaining)))

    # Ensure balanced labels in examples
    return selected[:n]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_eval(model: str, test_size: int = 50):
    print(f"\n{'='*65}")
    print(f"  Cybersec Threat Intel QA — Evaluation")
    print(f"  Model:     {model}")
    print(f"  Test size: {test_size}")
    print(f"  Few-shot:  {NUM_FEW_SHOT} examples")
    print(f"  Dataset:   {DATASET_PATH}")
    print(f"{'='*65}\n")

    # Load dataset
    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"[1/4] Loaded {len(data)} entries")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(data)
    test_set = data[:test_size]
    train_pool = data[test_size:]
    print(f"       Test:  {len(test_set)}")
    print(f"       Train: {len(train_pool)} (few-shot example pool)")

    # Label distribution in test set
    test_yes = sum(1 for e in test_set if str(e["label"]) == "1")
    test_no = len(test_set) - test_yes
    print(f"       Test labels: {test_yes} Yes / {test_no} No")

    # --- Zero-shot evaluation ---
    print(f"\n[2/4] Zero-shot evaluation...")
    zs_results = []
    zs_correct = 0
    zs_unknown = 0

    for i, entry in enumerate(test_set):
        prompt = zero_shot_prompt(entry["question"])
        response = query_ollama(prompt, model)
        prediction = parse_prediction(response)
        ground_truth = "Yes" if str(entry["label"]) == "1" else "No"
        correct = prediction == ground_truth

        if correct:
            zs_correct += 1
        if prediction == "Unknown":
            zs_unknown += 1

        zs_results.append({
            "question": entry["question"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": correct,
            "response": response[:500],
            "search_query": entry.get("search_query", ""),
        })

        status = "✓" if correct else ("?" if prediction == "Unknown" else "✗")
        print(f"  [{i+1:3d}/{test_size}] {status}  pred={prediction:<3s}  true={ground_truth:<3s}  {entry['question'][:80]}")

    zs_accuracy = zs_correct / len(test_set) * 100
    zs_valid = len(test_set) - zs_unknown
    zs_accuracy_valid = (zs_correct / zs_valid * 100) if zs_valid > 0 else 0
    print(f"\n  Zero-shot: {zs_correct}/{len(test_set)} = {zs_accuracy:.1f}% accuracy")
    if zs_unknown > 0:
        print(f"  ({zs_unknown} unparseable, {zs_accuracy_valid:.1f}% on parseable)")

    # --- Few-shot evaluation ---
    print(f"\n[3/4] Few-shot evaluation ({NUM_FEW_SHOT} examples per question)...")
    fs_results = []
    fs_correct = 0
    fs_unknown = 0

    for i, entry in enumerate(test_set):
        examples = select_few_shot_examples(train_pool, entry, NUM_FEW_SHOT)
        prompt = few_shot_prompt(entry["question"], examples)
        response = query_ollama(prompt, model, timeout=180)
        prediction = parse_prediction(response)
        ground_truth = "Yes" if str(entry["label"]) == "1" else "No"
        correct = prediction == ground_truth

        if correct:
            fs_correct += 1
        if prediction == "Unknown":
            fs_unknown += 1

        fs_results.append({
            "question": entry["question"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": correct,
            "response": response[:500],
            "search_query": entry.get("search_query", ""),
            "num_examples": len(examples),
        })

        status = "✓" if correct else ("?" if prediction == "Unknown" else "✗")
        print(f"  [{i+1:3d}/{test_size}] {status}  pred={prediction:<3s}  true={ground_truth:<3s}  {entry['question'][:80]}")

    fs_accuracy = fs_correct / len(test_set) * 100
    fs_valid = len(test_set) - fs_unknown
    fs_accuracy_valid = (fs_correct / fs_valid * 100) if fs_valid > 0 else 0
    print(f"\n  Few-shot: {fs_correct}/{len(test_set)} = {fs_accuracy:.1f}% accuracy")
    if fs_unknown > 0:
        print(f"  ({fs_unknown} unparseable, {fs_accuracy_valid:.1f}% on parseable)")

    # --- Per-category breakdown ---
    print(f"\n[4/4] Results by category:")
    categories = {}
    for zs, fs in zip(zs_results, fs_results):
        cat = zs["search_query"]
        if cat not in categories:
            categories[cat] = {"zs_correct": 0, "fs_correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if zs["correct"]:
            categories[cat]["zs_correct"] += 1
        if fs["correct"]:
            categories[cat]["fs_correct"] += 1

    print(f"  {'Category':<45s} {'ZS':>5s} {'FS':>5s} {'Δ':>5s}  n")
    print(f"  {'-'*70}")
    for cat, stats in sorted(categories.items(), key=lambda x: x[1]["total"], reverse=True):
        n = stats["total"]
        if n == 0:
            continue
        zs_pct = stats["zs_correct"] / n * 100
        fs_pct = stats["fs_correct"] / n * 100
        delta = fs_pct - zs_pct
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {cat:<45s} {zs_pct:4.0f}% {fs_pct:4.0f}% {arrow}{abs(delta):4.1f}  {n}")

    # --- Summary ---
    delta = fs_accuracy - zs_accuracy
    print(f"\n{'='*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"  Model:          {model}")
    print(f"  Test questions:  {len(test_set)}")
    print(f"  Zero-shot:       {zs_accuracy:.1f}% ({zs_correct}/{len(test_set)})")
    print(f"  Few-shot (5):    {fs_accuracy:.1f}% ({fs_correct}/{len(test_set)})")
    print(f"  Improvement:     {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"{'='*65}")

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "model": model,
            "test_size": len(test_set),
            "few_shot_examples": NUM_FEW_SHOT,
            "dataset": str(DATASET_PATH),
            "seed": SEED,
            "timestamp": timestamp,
        },
        "summary": {
            "zero_shot_accuracy": round(zs_accuracy, 1),
            "zero_shot_correct": zs_correct,
            "zero_shot_unknown": zs_unknown,
            "few_shot_accuracy": round(fs_accuracy, 1),
            "few_shot_correct": fs_correct,
            "few_shot_unknown": fs_unknown,
            "improvement": round(delta, 1),
            "test_label_balance": f"{test_yes} Yes / {test_no} No",
        },
        "per_category": {cat: stats for cat, stats in categories.items()},
        "zero_shot_results": zs_results,
        "few_shot_results": fs_results,
    }

    output_path = OUTPUT_DIR / f"eval_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cybersec QA dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--test-size", type=int, default=50, help="Number of test questions")
    args = parser.parse_args()

    run_eval(model=args.model, test_size=args.test_size)
