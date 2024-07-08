from collections import defaultdict, Counter
import argparse
from pathlib import Path
import pickle
from copy import deepcopy

import numpy as np
import torch

from nltk import Tree
from utils import clean_word, clean_symbol, tree_to_span


def main(
    filepath,
    vocab,
    output,
    binarization=False,
    horzMarkov=0,
    xbar=False,
    clean_label=False,
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
                factor="left",
                horzMarkov=horzMarkov,
                childChar="",
                parentChar="",
            )
            t.collapse_unary(collapsePOS=True)

    # Load vocab
    with open(vocab, "rb") as f:
        vocab = pickle.load(f)

    # Count unary productions
    length_count = defaultdict(int)
    tree_struct_by_length = defaultdict(lambda: defaultdict(int))
    tree_struct_by_length_n = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )

    for t in trees:
        length = len(t.leaves())
        if length <= 1:
            continue
        elif length > 40:
            continue

        length_count[length] += 1

        span = tree_to_span(t)
        span = [tuple(s[:2]) for s in span]
        span = tuple(
            filter(lambda s: not (s[0] == 0 and s[1] == length), span)
        )
        if clean_label:
            pt = tuple(
                [
                    clean_symbol(p[1].replace("'", ""), xbar=xbar)
                    for p in t.pos()
                ]
            )
        else:
            pt = tuple([p[1].replace("'", "") for p in t.pos()])

        # Count rules
        # tree_struct_by_length[length][span] += 1
        tree_struct_by_length_n[length][span][pt].add(tuple(t.leaves()))

    diversity_by_length = {
        length: diversity / length_count[length]
        for length, diversity in diversity_by_length.items()
    }
    diversity_by_length = dict(sorted(diversity_by_length.items()))

    # for length, counter in uniqueness_by_length.items():
    #     uniqueness_by_length[length] = dict(counter)
    #     print(f"Length: {length}, Unique rules: {len(counter)}")


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
    parser.add_argument("--clean_label", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.binarization,
        args.horzMarkov,
        args.xbar,
        args.clean_label,
    )
