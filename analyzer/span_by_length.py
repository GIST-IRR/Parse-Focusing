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
    label=False,
    clean_subtag=False,
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
    span_by_length = defaultdict(lambda: Counter())
    symbol_by_width = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for t in trees:
        length = len(t.leaves())
        if length <= 1:
            continue
        elif length > 40:
            continue

        span = tree_to_span(t)
        # Remove root span
        span = list(filter(lambda s: s[2] != "ROOT", span))
        # Remove trivial span
        span = list(filter(lambda s: (s[1] - s[0]) > 1, span))
        span = [
            (
                s[0],
                s[1],
                clean_symbol(s[2], xbar=xbar, cleanSubtag=clean_subtag),
            )
            for s in span
        ]

        # Count symbols by width
        for s in span:
            symbol_by_width[length][s[1] - s[0]][s[2]] += 1

        # Clean label
        if not label:
            span = [s[:2] for s in span]

        # Count rules
        span = Counter(span)
        span_by_length[length].update(span)

    # Sums
    symbol_by_width_sum = defaultdict(lambda: defaultdict(int))
    for length, d in symbol_by_width.items():
        for width, dd in d.items():
            for symbol, count in dd.items():
                symbol_by_width_sum[width][symbol] += count
    # Counts for width
    counts = {}
    for length, counter in span_by_length.items():
        tmp_c = defaultdict(int)
        for span, count in counter.items():
            tmp_c[span[1] - span[0]] += count
        tmp_c = sorted(tmp_c.items())
        counts[length] = tmp_c
    counts = sorted(counts.items())

    for length, counter in span_by_length.items():
        span_by_length[length] = dict(counter)
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
    parser.add_argument("--label", default=False, action="store_true")
    parser.add_argument("--clean_subtag", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.binarization,
        args.horzMarkov,
        args.xbar,
        args.label,
        args.clean_subtag,
    )
