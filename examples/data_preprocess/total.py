
import argparse
import os
import re, json

import datasets

from verl.utils.hdfs_io import copy, makedirs
from datasets import Dataset, DatasetDict, ClassLabel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/srv/nlprx-lab/share6/gguo37/rl/dynamic_mrpo_router_tuning/examples/data_preprocess/data/care_helpsteer_pku")

    args = parser.parse_args()
    
    with open('/srv/nlprx-lab/share5/gguo37/rl/data_analysis/region/final_dedu_safety.json', 'r', encoding='utf-8') as file:
        my_bench = json.load(file)
    full_ds = Dataset.from_list(my_bench)
    full_ds = full_ds.add_column("language_str", full_ds["language"])
    label_names = sorted(set(full_ds["language"]))
    full_ds = full_ds.cast_column("language", ClassLabel(names=label_names))

    split_ds = full_ds.train_test_split(test_size=0.1, shuffle=True, seed=42, stratify_by_column="language")

    dataset = DatasetDict(split_ds)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    data_source = "care_zh"

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["query"]
            chosen = example["chosen"]
            topic = example.pop("topic")
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": topic,
                "reward_model": {"ground_truth": chosen},
                "question": question,
                "answer": chosen,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "language": example['language_str'],
                    "topic": topic,
                    "region": example['region'],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

