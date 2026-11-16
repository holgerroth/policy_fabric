# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create deterministic, disjoint two-site splits of the official MSD BrainTumour data."""

import argparse
import json
import random
from pathlib import Path

DATA_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DATA_MD5 = "240a19d752f0d9e9101544901065d872"
SITES = ("site_a", "site_b")


def prepare(dataset_root, output, train_per_site=4, val_per_site=2, seed=2026):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    metadata = json.loads((dataset_root / "dataset.json").read_text())
    expected_modalities = {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}
    if {k: v.lower() for k, v in metadata["modality"].items()} != {
        k: v.lower() for k, v in expected_modalities.items()
    }:
        raise ValueError(f"Unexpected MSD channel order: {metadata['modality']}")
    expected_labels = {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    if metadata["labels"] != expected_labels:
        raise ValueError(f"Unexpected MSD label meanings: {metadata['labels']}")
    records = sorted(metadata["training"], key=lambda item: item["image"])
    if train_per_site < 0 or val_per_site < 1:
        raise ValueError("train_per_site must be nonnegative; val_per_site must be positive")
    if len({item["image"] for item in records}) != len(records):
        raise ValueError("Duplicate cases in dataset metadata")
    if len({item["label"] for item in records}) != len(records):
        raise ValueError("Duplicate labels in dataset metadata")
    random.Random(seed).shuffle(records)
    val_count = len(SITES) * val_per_site
    if train_per_site == 0:
        train_per_site = (len(records) - val_count) // len(SITES)
    count = len(SITES) * train_per_site + val_count
    if train_per_site < 1 or count > len(records):
        raise ValueError("Not enough cases for the requested disjoint split")
    selected = records[:count]
    for item in selected:
        for key in ("image", "label"):
            path = (dataset_root / item[key]).resolve()
            if not path.is_relative_to(dataset_root) or not path.is_file():
                raise ValueError(f"Missing or invalid dataset path: {item[key]}")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "dataset_root": str(dataset_root),
        "dataset": metadata["name"],
        "dataset_source": DATA_URL,
        "archive_md5": DATA_MD5,
        "available_labeled_cases": len(records),
        "seed": seed,
        "input_modalities": metadata["modality"],
        "input_labels": metadata["labels"],
        "bundle_channel_indices": [2, 1, 3, 0],
        "output_regions": {"TC": [2, 3], "WT": [1, 2, 3], "ET": [3]},
        "sites": {},
    }
    validation = records[:val_count]
    training = records[val_count:count]
    for index, site in enumerate(SITES):
        split = {
            "training": training[index :: len(SITES)],
            "validation": validation[index :: len(SITES)],
        }
        (output / f"{site}.json").write_text(json.dumps(split, indent=2) + "\n")
        summary["sites"][site] = {
            key: {"count": len(value), "cases": [Path(item["image"]).name for item in value]}
            for key, value in split.items()
        }
    (output / "evaluation.json").write_text(json.dumps({"validation": validation}, indent=2) + "\n")
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-per-site", type=int, default=4, help="0 uses all remaining cases, balanced across sites"
    )
    parser.add_argument("--val-per-site", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--download", action="store_true", help="Download and checksum the official archive if needed")
    args = parser.parse_args()
    if args.download:
        from monai.apps.utils import download_and_extract

        if args.dataset_root.name != "Task01_BrainTumour":
            parser.error("--download requires dataset-root ending in Task01_BrainTumour")
        args.dataset_root.parent.mkdir(parents=True, exist_ok=True)
        download_and_extract(
            DATA_URL, args.dataset_root.with_suffix(".tar"), args.dataset_root.parent, hash_val=DATA_MD5
        )
    print(
        json.dumps(prepare(args.dataset_root, args.output, args.train_per_site, args.val_per_site, args.seed), indent=2)
    )
