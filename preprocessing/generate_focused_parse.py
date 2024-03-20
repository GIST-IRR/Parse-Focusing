import pickle
from copy import deepcopy
import argparse
from pathlib import Path

import numpy as np
import torch
from nltk import Tree

from utils import sort_span, clean_word, tree_to_span


def random_split(start, end):
    if end - start == 1:
        return []
    else:
        spans = [(start, end)]
        split = np.random.randint(start + 1, end)
        spans += random_split(start, split)
        spans += random_split(split, end)
        return spans


def branching(len, factor="left-branching"):
    if factor == "left-branching":
        trees = [(0, len - i) for i in range(len - 1)]
    elif factor == "right-branching":
        trees = [(i, len) for i in range(len - 1)]
    elif factor == "random":
        trees = random_split(0, len)
        trees = sort_span([trees])[0]
    return trees


def main(factor, input_path, output_path, vocab_path):
    with Path(vocab_path).open("rb") as f:
        vocab = pickle.load(f)

    pts = []
    with Path(input_path).open("r") as f:
        for l in f:
            t = Tree.fromstring(l)
            pos_word = t.pos()
            sentence, _ = zip(*pos_word)
            word = [vocab[w] for w in clean_word(sentence)]

            if (
                factor == "left-branching"
                or factor == "right-branching"
                or factor == "random"
            ):
                length = len(sentence)
                tree = branching(length, factor)
            elif factor == "left-binarized" or factor == "right-binarized":
                tree = deepcopy(t)
                tree.collapse_unary(collapsePOS=True)
                tree.chomsky_normal_form(factor=factor.split("-")[0])
                tree = tree_to_span(tree)
                tree = [tuple(s) for s in tree]
            elif factor == "gold":
                tree = tree_to_span(t)
                tree = [tuple(s) for s in tree]

            pt = {"sentence": sentence, "word": word, "tree": tree}
            pts.append(pt)

    torch.save(pts, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=str, default="right-binarized")
    parser.add_argument("--vocab", type=str, default="vocab/chinese.vocab")
    parser.add_argument(
        "--input", type=str, default="data/data.clean/edit-chinese-train.txt"
    )
    parser.add_argument(
        "--output", type=str, default="trees/right-binarized_testword_train.pt"
    )
    args = parser.parse_args()
    main(args.factor, args.input, args.output, args.vocab)
