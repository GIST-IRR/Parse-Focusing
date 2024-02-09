#!/usr/bin/env python
import pickle
import argparse
from pathlib import Path
from collections import OrderedDict, Counter, defaultdict
import random
from numpy import vsplit
from torch.utils.tensorboard import SummaryWriter

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
import torch
from utils import span_to_tree

from parser.helper.metric import UF1


def split_by_key(data, key):
    # split data to ordered dictionary by key
    result = {}
    for d in data:
        v = d[key]
        if v in result:
            result[v].append(d)
        else:
            result[v] = [d]
    return OrderedDict(sorted(result.items()))


def count_value(data, key):
    result = {}
    for d in data:
        n = d[key]
        if n in result:
            result[n] += 1
        else:
            result[n] = 1
    return OrderedDict(sorted(result.items()))


def main(args, noise_count=1, noise_threshold=0.1):
    data_name = args.dataset
    with open(data_name, "rb") as f:
        dataset = pickle.load(f)

    pos_tags = {t for ts in dataset["pos"] for t in ts}
    words = {w for ws in dataset["word"] for w in ws}
    words_by_tags = defaultdict(set)
    for sent, tags in zip(dataset["word"], dataset["pos"]):
        for word, tag in zip(sent, tags):
            words_by_tags[tag].add(word.lower())
    words_by_tags = {k: list(v) for k, v in words_by_tags.items()}

    dataset_by_idx = []
    noised_dataset_by_idx = []
    noised_oneline = []
    for w, p, gt, gtl, gtr, d, dl, dr in zip(
        dataset["word"],
        dataset["pos"],
        dataset["gold_tree"],
        dataset["gold_tree_left"],
        dataset["gold_tree_right"],
        dataset["depth"],
        dataset["depth_left"],
        dataset["depth_right"],
    ):
        dataset_by_idx.append(
            {
                "word": w,
                "pos": p,
                "gold_tree": gt,
                "gold_tree_left": gtl,
                "gold_tree_right": gtr,
                "depth": d,
                "depth_left": dl,
                "depth_right": dr,
            }
        )

        count = 0
        while count < noise_count:
            k = int(len(w) * noise_threshold)
            k = max(k, 1)

            word_idx = random.sample(range(len(w)), k=k)
            word_tag = [p[i] for i in word_idx]
            new_words = [random.choice(words_by_tags[t]) for t in word_tag]
            n_sent = list(w)
            for i, nw in zip(word_idx, new_words):
                n_sent[i] = nw

            if n_sent != w:
                noised_dataset_by_idx.append(
                    {
                        "word": n_sent,
                        "pos": p,
                        "gold_tree": gt,
                        "gold_tree_left": gtl,
                        "gold_tree_right": gtr,
                        "depth": d,
                        "depth_left": dl,
                        "depth_right": dr,
                    }
                )
                noised_tree = span_to_tree(gt)
                for i, tree in enumerate(
                    list(noised_tree.subtrees(lambda t: t.height() == 2))
                ):
                    tree.set_label(p[i])
                    tree[0] = n_sent[i]
                noised_oneline.append(
                    noised_tree._pformat_flat("", "()", False)
                )
                count += 1

    noised_dataset = defaultdict(list)
    for i in noised_dataset_by_idx:
        noised_dataset["word"].append(i["word"])
        noised_dataset["pos"].append(i["pos"])
        noised_dataset["gold_tree"].append(i["gold_tree"])
        noised_dataset["gold_tree_left"].append(i["gold_tree_left"])
        noised_dataset["gold_tree_right"].append(i["gold_tree_right"])
        noised_dataset["depth"].append(i["depth"])
        noised_dataset["depth_left"].append(i["depth_left"])
        noised_dataset["depth_right"].append(i["depth_right"])

    data_path = Path(data_name)
    dir_path = data_path.parent
    data_name = data_path.stem
    with open((dir_path / data_name).with_suffix(".noised.pkl"), "wb") as f:
        pickle.dump(noised_dataset, f)

    with open(
        (dir_path / data_name).with_suffix(
            f".{noise_count}.{noise_threshold}.noised.txt"
        ),
        "w",
    ) as f:
        f.write("\n".join(noised_oneline))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise_count", "-c", type=int, default=1)
    parser.add_argument("--noise_threshold", "-t", type=float, default=0.1)
    args = parser.parse_args()

    main(args, args.noise_count, args.noise_threshold)
