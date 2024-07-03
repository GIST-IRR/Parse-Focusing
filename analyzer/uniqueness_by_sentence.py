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
    uniqueness_by_length = defaultdict(lambda: Counter())

    for t in trees:
        length = len(t.leaves())
        if length <= 1:
            continue
        elif length > 40:
            continue

        # Get productions
        prod = [
            p
            for p in t.productions()
            # if not isinstance(p.rhs()[0], str) and p.lhs().symbol() != "ROOT"
            if not isinstance(p.rhs()[0], str)
        ]

        # Clean symbol
        def clean_rule(rule):
            n_rule = deepcopy(rule)
            lhs = clean_symbol(rule.lhs().symbol(), xbar=xbar)
            rhs = [clean_symbol(r.symbol(), xbar=xbar) for r in rule.rhs()]

            n_rule.lhs()._symbol = lhs
            for i, r in enumerate(n_rule.rhs()):
                r._symbol = rhs[i]
            return n_rule

        # prod = list(map(clean_rule, prod))

        # Count rules
        prod = Counter(prod)
        uniqueness_by_length[length].update(prod)

    # Counter to dict
    n_uniqueness_by_length = {}
    for length, counter in sorted(uniqueness_by_length.items()):
        total_count = counter.total()
        mass = 0
        n_dict = {}
        c = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for rule, count in c:
            count /= total_count
            mass += count
            if mass >= 0.95:
                break
            n_dict[rule] = count
        n_uniqueness_by_length[length] = n_dict

    rule_types_for_length = {
        k: set(list(v.keys())) for k, v in uniqueness_by_length.items()
    }
    total_rules = set.union(*list(rule_types_for_length.values()))

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
