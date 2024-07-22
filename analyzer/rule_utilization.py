from collections import defaultdict, Counter
import argparse
from pathlib import Path
import pickle
from copy import deepcopy

import numpy as np
import torch

from nltk import Tree
from utils import clean_symbol, clean_word


def main(
    filepath,
    vocab,
    output,
    binarization=False,
    horzMarkov=0,
    xbar=False,
    clean_label=False,
    clean_subtag=False,
    min_len=2,
    max_len=40,
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
    rule_utilization = defaultdict(lambda: Counter())
    term_rule_util = defaultdict(lambda: Counter())

    for t in trees:
        length = len(t.leaves())
        if length < min_len:
            continue
        elif length > max_len:
            continue

        # Get productions
        prod = [
            p
            for p in t.productions()
            if not isinstance(p.rhs()[0], str) and p.lhs().symbol() != "ROOT"
            # if not isinstance(p.rhs()[0], str)
        ]

        term_prod = [p for p in t.productions() if isinstance(p.rhs()[0], str)]

        # Clean symbol
        def clean_rule(rule):
            n_rule = deepcopy(rule)
            lhs = clean_symbol(
                rule.lhs().symbol(), xbar=xbar, cleanSubtag=clean_subtag
            )
            rhs = [
                clean_symbol(r.symbol(), xbar=xbar, cleanSubtag=clean_subtag)
                for r in rule.rhs()
            ]

            n_rule.lhs()._symbol = lhs
            for i, r in enumerate(n_rule.rhs()):
                r._symbol = rhs[i]
            return n_rule

        def clean_unary(rule):
            n_rule = deepcopy(rule)
            lhs = clean_symbol(
                rule.lhs().symbol(), xbar=xbar, cleanSubtag=clean_subtag
            )
            rhs = vocab.to_index(clean_word(rule.rhs())[0])
            rhs = vocab.to_word(rhs)
            n_rule.lhs()._symbol = lhs
            n_rule._rhs = (rhs,)
            return n_rule

        if clean_label:
            prod = list(map(clean_rule, prod))
            term_prod = list(map(clean_unary, term_prod))

        # Count rules
        # prod = set(prod)
        rule_utilization[length].update(prod)
        for p in term_prod:
            term_rule_util[p.lhs()._symbol].update([p.rhs()[0]])

    term_util_prob = defaultdict(lambda: defaultdict())
    for length, counter in term_rule_util.items():
        total = counter.total()
        for term, count in counter.items():
            term_util_prob[length][term] = count / total

    total_ru = defaultdict(int)
    for l, d in rule_utilization.items():
        for r, v in d.items():
            total_ru[r] += v
    total_ru = {
        k: v
        for k, v in sorted(total_ru.items(), key=lambda x: x[1], reverse=True)
    }
    total_ru_by_symbol = defaultdict(
        lambda: defaultdict(lambda: defaultdict())
    )
    for r, v in total_ru.items():
        lhs = r.lhs()._symbol
        rhs_0 = r.rhs()[0]._symbol
        rhs_1 = r.rhs()[1]._symbol
        total_ru_by_symbol[lhs][rhs_0][rhs_1] = v

    torch.save(total_ru, f"rule_utilization/{Path(filepath).stem}.pt")
    # Counter to dict
    # n_uniqueness_by_length = {}
    # for length, counter in sorted(uniqueness_by_length.items()):
    #     total_count = counter.total()
    #     mass = 0
    #     n_dict = {}
    #     c = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    #     for rule, count in c:
    #         count /= total_count
    #         mass += count
    #         if mass >= 0.95:
    #             break
    #         n_dict[rule] = count
    #     n_uniqueness_by_length[length] = n_dict

    # for length, counter in uniqueness_by_length.items():
    #     uniqueness_by_length[length] = dict(counter)
    #     print(f"Length: {length}, Unique rules: {len(counter)}")


if __name__ == "__main__":
    total_ru_by_symbol = defaultdict(
        lambda: defaultdict(lambda: defaultdict())
    )
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
    parser.add_argument("--clean_subtag", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.binarization,
        args.horzMarkov,
        args.xbar,
        args.clean_label,
        args.clean_subtag,
    )
