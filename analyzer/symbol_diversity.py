from collections import defaultdict, Counter
import argparse
from pathlib import Path
import pickle
from copy import deepcopy

import numpy as np
import torch

from nltk import Tree
from utils import clean_word, clean_symbol


def main(
    filepath,
    vocab,
    output,
    binarization=False,
    horzMarkov=0,
    xbar=False,
):
    # Load trees
    trees = []
    with open(filepath, "r") as f:
        for l in f:
            trees.append(Tree.fromstring(l))
    # Tree binarization
    if binarization:
        for t in trees:
            t.chomsky_normal_form(
                horzMarkov=horzMarkov, childChar="", parentChar=""
            )
            t.collapse_unary(collapsePOS=True)

    # Load vocab
    with open(vocab, "rb") as f:
        vocab = pickle.load(f)

    # Count unary productions
    length_count = defaultdict(int)
    uniqueness_by_length = defaultdict(lambda: Counter())
    diversity_by_length = defaultdict(int)
    pos_div_by_length = defaultdict(int)

    for t in trees:
        length = len(t.leaves())
        if length <= 1:
            continue
        elif length > 40:
            continue

        length_count[length] += 1

        position = t.treepositions()
        labels = [t[p].label() for p in position if not isinstance(t[p], str)]

        pos_tag = set([p[1] for p in t.pos()])
        pos_div = len(pos_tag)
        pos_div_by_length[length] += pos_div

        # prod = list(map(clean_rule, prod))
        # labels = list(
        #     map(
        #         clean_symbol,
        #         labels,
        #         [False] * len(labels),
        #         [xbar] * len(labels),
        #     )
        # )

        # Count rules
        labels = Counter(labels)
        diversity = len(labels)
        diversity_by_length[length] += diversity

    pos_div_by_length = {
        length: diversity / length_count[length]
        for length, diversity in pos_div_by_length.items()
    }
    pos_div_by_length = dict(sorted(pos_div_by_length.items()))

    diversity_by_length = {
        length: diversity / length_count[length]
        for length, diversity in diversity_by_length.items()
    }
    diversity_by_length = dict(sorted(diversity_by_length.items()))

    for length, counter in uniqueness_by_length.items():
        uniqueness_by_length[length] = dict(counter)
        print(f"Length: {length}, Unique rules: {len(counter)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/data.clean/edit-english-train.txt",
    )
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument("--output", type=str, default="gold_term_prob.pkl")
    parser.add_argument("--binarization", default=False, action="store_true")
    parser.add_argument("--horzMarkov", type=int, default=0)
    parser.add_argument("--xbar", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.binarization,
        args.horzMarkov,
        args.xbar,
    )
