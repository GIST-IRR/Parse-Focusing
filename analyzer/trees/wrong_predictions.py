import pickle
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import itertools

import numpy as np
import torch
from nltk import Tree

from utils import load_trees, span_to_tree, tree_to_span, sort_span, clean_word
from torch_support.metric import preprocess_span
from torch_support.metric import sentence_level_f1 as f1
from torch_support.metric import iou as IoU


def set_leaves(tree, words):
    tp = tree.treepositions("leaves")
    for i, p in enumerate(tp):
        tree[p] = str(words[i])


def production_by_span(tree, span):
    start, end = span
    start = tree.leaf_treeposition(start)
    end = tree.leaf_treeposition(end - 1)

    split = []
    for s, e in zip(start, end):
        if s != e:
            break
        else:
            split.append(s)

    if len(split) > 0:
        t = tree[*split]
    else:
        t = tree

    return t.productions()[0]


def pretty_print(tree):
    word = tree["word"]

    def _pretty_print(tree, word):
        t = span_to_tree(tree)
        set_leaves(t, word)
        t.pretty_print()

    _pretty_print(tree["gold"], word)
    _pretty_print(tree["pred1"], word)
    _pretty_print(tree["pred2"], word)


def main(pred, gold, vocab, max_len=40, min_len=2):
    # Load vocab
    if vocab is not None:
        with vocab.open("rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = None
        raise ValueError("Vocab is required.")

    # Load gold parse trees
    if gold is not None:
        args.eval_gold = True
        gold = load_trees(gold, min_len, max_len, True, vocab)

    # Load predicted parse tree
    trees = load_trees(pred, min_len, max_len, True, vocab)

    # Container
    fp_prods = Counter()
    fn_prods = Counter()
    match_up = []

    for i, t in enumerate(trees):
        word = t["word"]
        sentence = t["sentence"]
        length = len(word)

        # Preprocess gold trees
        if args.eval_gold:
            gs = gold[i]
            # Check predicted trees and gold tree are same
            assert gs["sentence"] == sentence

            gt = gs["span"]
            gt = [s[:2] for s in gt]
            gt = sort_span(gt)

        # Preprocessing predicted trees
        pd = [s[:2] for s in t["span"]]
        # Add root and trivial spans
        # if len(pd) < len(word):
        #     pd.append((0, len(word)))
        #     for i in range(len(word)):
        #         pd.append((i, i + 1))
        pd = sort_span(pd)

        gt = set(preprocess_span(gt, length))
        pd = set(preprocess_span(pd, length))
        # False Positive
        fp = pd - gt
        fp_prod = tuple([production_by_span(t["tree"], s) for s in fp])
        fp_prods.update(fp_prod)
        # False Negative
        fn = gt - pd
        fn_prod = tuple([production_by_span(gs["tree"], s) for s in fn])
        fn_prods.update(fn_prod)
        if len(fp) > 0 or len(fn) > 0:
            match_up.append({"fp": fp_prod, "fn": fn_prod})
        # print("test")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--pred", required=True, type=Path)
    parser.add_argument("-g", "--gold", required=True, type=Path)
    parser.add_argument("-v", "--vocab", required=True, type=Path)
    args = parser.parse_args()

    main(args.pred, args.gold, args.vocab)
