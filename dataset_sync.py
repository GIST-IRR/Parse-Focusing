#!/usr/bin/env python
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def dataset_sync(datasets_path, output_path):
    # Open datasets
    datasets = [pickle.load(open(d, "rb")) for d in datasets_path]
    # Add attribute "length" to each dataset
    for d in datasets:
        d["length"] = [len(w) for w in list(d.values())[0]]
    # Count length distribution
    counters = defaultdict(list)
    for d in datasets:
        c = Counter(d["length"])
        for k, v in c.items():
            counters[k].append(v)

    # Find the intersection of length distribution
    int_count = {}
    for k, v in counters.items():
        if len(v) != len(datasets):
            continue
        v = min(v)
        int_count[k] = v

    # Reconstruct datasets by the intersection of length distribution
    new_datasets = []
    for d in datasets:
        count = int_count.copy()
        n_dataset = {}
        for i, l in enumerate(d["length"]):
            if l in count and count[l] > 0:
                for k, v in d.items():
                    if k not in n_dataset:
                        n_dataset[k] = []
                    n_dataset[k].append(v[i])
                count[l] -= 1
        new_datasets.append(n_dataset)

    # Sort datasets by length
    for d in new_datasets:
        sort_idx = np.argsort(d["length"])
        for k, v in d.items():
            d[k] = [v[i] for i in sort_idx]

    # Save datasets
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for i, d in enumerate(new_datasets):
        tag = datasets_path[i].split("/")[-1].split(".")[0]
        with (output_path / f"sync_{tag}.pkl").open("wb") as f:
            pickle.dump(d, f)

    print("New datasets are saved in", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_sync(args.datasets, args.output)
