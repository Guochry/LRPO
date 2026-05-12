
import argparse
import os
import re, json

import datasets

from verl.utils.hdfs_io import copy, makedirs
from datasets import Dataset, DatasetDict, ClassLabel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/srv/nlprx-lab/share6/gguo37/culture_adapt/verl/data/CARE")

    args = parser.parse_args()
    
    with open('/srv/nlprx-lab/share6/gguo37/rl/mrmbench/data/mrewardbench_aligned_en_ar_ja_zh.json', 'r', encoding='utf-8') as file:
        my_bench = json.load(file)
    full_ds = Dataset.from_list(my_bench)
    split_ds = full_ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

    dataset = DatasetDict(split_ds)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    # dataset = DatasetDict({"train": full_ds})
    # train_dataset = dataset["train"]
    # print(len(train_dataset))

    data_source = "mrm_aligned"

    def make_map_fn(split):
        def process_fn(example, idx):
            question_en = example.pop("question_en")
            answer_en = example.pop("answer_en")
            question_ar = example.pop("question_ar")
            answer_ar = example.pop("answer_ar")
            question_ja = example.pop("question_ja")
            answer_ja = example.pop("answer_ja")
            question_zh = example.pop("question_zh")
            answer_zh = example.pop("answer_zh")
            category = example.pop("category")
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question_en}],
                "reward_model": {"ground_truth": answer_en},
                "ability": category,
                "question_en": question_en,
                "answer_en": answer_en,
                "question_ar": question_ar,
                "answer_ar": answer_ar,
                "question_ja": question_ja,
                "answer_ja": answer_ja,
                "question_zh": question_zh,
                "answer_zh": answer_zh,
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

