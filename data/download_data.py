import argparse
from huggingface_hub import hf_hub_download
import shutil
import pandas as pd


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["PrimeIntellect","MATH_train","demon_openr1math","MATH500","GSM8K","AIME2024","LiveBench","LiveCodeBench","MBPP","HumanEval","MMLU-Pro"],
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
else:
    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{dataset}",
        repo_type="dataset",
        filename=f"{split}/{dataset}.json"
    )
    shutil.copy(cached_path, f"./{dataset}.json")