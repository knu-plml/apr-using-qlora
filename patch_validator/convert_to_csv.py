#!/usr/bin/env python3
"""
JSON to CSV Converter for Plausible Patch Validation

This script converts JSON validation data from multiple model folders
into a single CSV file for human evaluation.
"""

import json
import csv
import os
import re
from pathlib import Path


DATA_DIR = "/home/yglee/wlm"
OUTPUT_CSV = "/home/yglee/wl/p14/patch_validator/evaluations.csv"

# Benchmark configurations
# For defects4j, we use _validate_v12 and _validate_v20 instead of base _validate
BENCHMARKS = {
    # "defects4j_v12": {
    #     "input_file": "defects4j_finetune_input.json",
    #     "validate_file": "defects4j_finetune_strict_validate_v12.json",
    # },
    # "defects4j_v20": {
    #     "input_file": "defects4j_finetune_input.json",
    #     "validate_file": "defects4j_finetune_strict_validate_v20.json",
    # },
    # "humaneval": {
    #     "input_file": "humaneval_finetune_input.json",
    #     "validate_file": "humaneval_finetune_validate.json",
    # },
    "quixbugs": {
        "input_file": "quixbugs_finetune_input.json",
        "validate_file": "quixbugs_finetune_validate.json",
    },
}


def get_model_folders(data_dir: str) -> list[str]:
    """Get all model folders matching the pattern [model]_v[number]"""
    pattern = re.compile(r'^.+_v\d+$')
    folders = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path) and pattern.match(name):
            folders.append(name)
    return sorted(folders)


def load_json(filepath: str) -> dict | None:
    """Load a JSON file, return None if not exists"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_plausible_patches(data_dir: str, model: str, benchmark: str, config: dict) -> list[dict]:
    """Extract all plausible patches from a model's benchmark validation file"""
    results = []

    model_path = os.path.join(data_dir, model)
    input_path = os.path.join(model_path, config["input_file"])
    validate_path = os.path.join(model_path, config["validate_file"])

    # Load input data for fixed_line
    input_data = load_json(input_path)
    if input_data is None:
        return results

    # Load validation data
    validate_data = load_json(validate_path)
    if validate_data is None:
        return results

    input_dict = input_data.get("data", {})
    validate_dict = validate_data.get("data", {})

    for problem_id, validate_item in validate_dict.items():
        outputs = validate_item.get("output", [])
        input_text = validate_item.get("input", "")

        # Get fixed_line from input data
        fixed_line = ""
        if problem_id in input_dict:
            fixed_line = input_dict[problem_id].get("fixed_line", "")

        for patch_index, output_item in enumerate(outputs):
            correctness = output_item.get("correctness", "")
            if correctness == "plausible":
                results.append({
                    "problem_id": problem_id,
                    "benchmark": benchmark,
                    "model": model,
                    "patch_index": patch_index,
                    "patch": output_item.get("patch", ""),
                    "input": input_text,
                    "fixed_line": fixed_line,
                    "evaluation": "",  # To be filled by human evaluator
                })

    return results


def main():
    print(f"Scanning model folders in {DATA_DIR}...")
    model_folders = get_model_folders(DATA_DIR)
    print(f"Found {len(model_folders)} model folders")

    all_patches = []

    for model in model_folders:
        print(f"Processing {model}...")
        for benchmark, config in BENCHMARKS.items():
            patches = extract_plausible_patches(DATA_DIR, model, benchmark, config)
            all_patches.extend(patches)
            if patches:
                print(f"  {benchmark}: {len(patches)} plausible patches")

    # Sort by problem_id, then benchmark, then model, then patch_index
    all_patches.sort(key=lambda x: (x["problem_id"], x["benchmark"], x["model"], x["patch_index"]))

    # Write to CSV
    print(f"\nWriting {len(all_patches)} patches to {OUTPUT_CSV}...")

    fieldnames = ["problem_id", "benchmark", "model", "patch_index", "patch", "input", "fixed_line", "evaluation"]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_patches)

    print("Done!")

    # Print summary
    print("\n=== Summary ===")
    benchmark_counts = {}
    model_counts = {}
    problem_counts = set()

    for p in all_patches:
        benchmark_counts[p["benchmark"]] = benchmark_counts.get(p["benchmark"], 0) + 1
        model_counts[p["model"]] = model_counts.get(p["model"], 0) + 1
        problem_counts.add(p["problem_id"])

    print(f"Total plausible patches: {len(all_patches)}")
    print(f"Unique problems: {len(problem_counts)}")
    print(f"\nBy benchmark:")
    for b, c in sorted(benchmark_counts.items()):
        print(f"  {b}: {c}")
    print(f"\nBy model:")
    for m, c in sorted(model_counts.items()):
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
