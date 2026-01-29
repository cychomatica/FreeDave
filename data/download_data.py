import argparse
from huggingface_hub import hf_hub_download
import shutil
import pandas as pd
import json
import os


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["PrimeIntellect","MATH_train","demon_openr1math","MATH500","GSM8K","AIME2024","AIME2025","LiveBench","LiveCodeBench","MBPP","HumanEval","MMLU-Pro"],
    required=True,
    help="Which dataset to download"
)
args = parser.parse_args()
dataset = args.dataset


if dataset == "MATH_train" or dataset == "PrimeIntellect" or dataset == "demon_openr1math":
    split = "train"
else:
    split = "test"

if dataset == "MMLU-Pro":
    cached_path = hf_hub_download(
        repo_id=f"TIGER-Lab/MMLU-Pro",
        repo_type="dataset",
        filename=f"data/{split}-00000-of-00001.parquet"
    )
    df = pd.read_parquet(cached_path)
    df.to_json(f"./{dataset}.json", orient="records", indent=4)
elif dataset == "AIME2025":
    # Download both AIME2025-I and AIME2025-II files
    cached_path_1 = hf_hub_download(
        repo_id="opencompass/AIME2025",
        repo_type="dataset",
        filename="aime2025-I.jsonl"
    )
    cached_path_2 = hf_hub_download(
        repo_id="opencompass/AIME2025",
        repo_type="dataset",
        filename="aime2025-II.jsonl"
    )
    # Read and merge both JSONL files
    merged_data = []
    for cached_path in [cached_path_1, cached_path_2]:
        with open(cached_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Rename "answer" to "ground_truth_answer" for consistency with AIME2024
                    if "answer" in item and "ground_truth_answer" not in item:
                        item["ground_truth_answer"] = item.pop("answer")
                    merged_data.append(item)
    # Save merged data as JSON
    with open(f"./{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    # Delete original cached files
    os.remove(cached_path_1)
    os.remove(cached_path_2)
else:
    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{dataset}",
        repo_type="dataset",
        filename=f"{split}/{dataset}.json"
    )
    shutil.copy(cached_path, f"./{dataset}.json")