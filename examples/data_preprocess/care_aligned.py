import argparse
import os
import json
from collections import Counter
from datasets import (
    Dataset,
    DatasetDict,
    ClassLabel,
    concatenate_datasets,
)
from verl.utils.hdfs_io import copy, makedirs

JSON_PATHS = [
    "/srv/nlprx-lab/share6/gguo37/rl/x-care/data-rft/train/care_ar_aligned.json",
    "/srv/nlprx-lab/share6/gguo37/rl/x-care/data-rft/train/care_ja_aligned.json",
    "/srv/nlprx-lab/share6/gguo37/rl/x-care/data-rft/train/care_zh_aligned.json",
]
TEST_RATIO   = 0.1
SEED         = 42
DATA_SOURCE  = "care"

def collect_labels_and_raw(paths):
    all_labels = set()
    raw_data   = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            records = json.load(f)
        raw_data[p] = records
        all_labels.update(r["cul_type"] for r in records)
    return sorted(all_labels), raw_data

def split_with_rare_handling(records, label_feature):
    ds_full = Dataset.from_list(records).cast_column("cul_type", label_feature)

    cul_counts = Counter(ds_full["cul_type"])
    rare_idx   = [i for i, c in enumerate(ds_full["cul_type"]) if cul_counts[c] < 2]
    rest_idx   = [i for i in range(len(ds_full)) if i not in rare_idx]

    rare_ds = ds_full.select(rare_idx)         if rare_idx else None
    rest_ds = ds_full.select(rest_idx)         if rest_idx else None

    if rest_ds is None:
        train_final = rare_ds
        test_final  = ds_full.select([])
    else:
        split_rest = rest_ds.train_test_split(
            test_size=TEST_RATIO,
            seed=SEED,
            stratify_by_column="cul_type"
        )
        train_final = concatenate_datasets(
            [split_rest["train"], rare_ds] if rare_ds else [split_rest["train"]]
        )
        test_final  = split_rest["test"]

    return train_final, test_final

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
        return {
            "data_source": DATA_SOURCE,
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
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="/srv/nlprx-lab/share6/gguo37/culture_adapt/verl/data/CARE",
        help="Where to save parquet files"
    )
    args = parser.parse_args()

    all_labels, raw_data = collect_labels_and_raw(JSON_PATHS)
    label_feature = ClassLabel(names=all_labels)

    train_parts, test_parts = [], []
    for recs in raw_data.values():
        tr, te = split_with_rare_handling(recs, label_feature)
        train_parts.append(tr)
        test_parts.append(te)

    train_dataset = concatenate_datasets(train_parts)
    test_dataset  = concatenate_datasets(test_parts)

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    test_dataset  = test_dataset.map(make_map_fn("test"),  with_indices=True)

    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    print("Done!  Train =", len(train_dataset), "samples ;  Test =", len(test_dataset))
