import pickle
from copy import deepcopy
import argparse
from pathlib import Path

import numpy as np
import torch
from nltk import Tree

from utils import sort_span, clean_word, tree_to_span, clean_symbol


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
        trees = sort_span(trees)
    return trees


def main(factor, input_path, output_path, vocab_path, xbar, joinChar):
    with Path(vocab_path).open("rb") as f:
        vocab = pickle.load(f)

    pts = []
    trees = []
    input_path = Path(input_path)
    with input_path.open("r") as f:
        for l in f:
            t = Tree.fromstring(l)
            pos_word = t.pos()
            sentence, pos = zip(*pos_word)
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
                tree.collapse_unary(collapsePOS=True, joinChar=joinChar)
                tree.chomsky_normal_form(
                    factor=factor.split("-")[0],
                    horzMarkov=0,
                    childChar="",
                    parentChar="",
                )
                org_tree = tree
                tree = tree_to_span(tree)
                # remove root
                tree = list(filter(lambda x: x[2] != "ROOT", tree))
                # Extract labels
                label = [
                    clean_symbol(
                        s[2], xbar=xbar, cleanSubtag=True, joinChar=joinChar
                    )
                    for s in tree
                ]
                # Tuple
                tree = [tuple(s) for s in tree]
                # org_tree change symbols
                for t in org_tree.subtrees():
                    if t.label() != "ROOT":
                        t.set_label(
                            clean_symbol(
                                t.label(),
                                xbar=xbar,
                                cleanSubtag=False,
                                joinChar=joinChar,
                            )
                        )
                trees.append(org_tree._pformat_flat("", "()", False))
            elif factor == "gold":
                tree = tree_to_span(t)
                # remove root
                tree = list(filter(lambda x: x[2] != "ROOT", tree))
                # Tuple
                tree = [tuple(s) for s in tree]

            pt = {"sentence": sentence, "word": word, "tree": tree, "pos": pos}
            try:
                pt.update({"label": label})
            except:
                pass

            pts.append(pt)

    tree_path = Path(f"parsed/{Path(output_path).stem}.txt")
    tree_path.write_text("\n".join(trees))
    # torch.save(pts, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=str, default="right-binarized")
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument(
        "--input", type=str, default="data/data.clean/edit-english-test.txt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trees/right-binarized-label_english_test.pt",
    )
    parser.add_argument(
        "--xbar",
        action="store_true",
    )
    parser.add_argument(
        "--joinChar",
        type=str,
        default="+",
    )
    args = parser.parse_args()
    main(
        args.factor,
        args.input,
        args.output,
        args.vocab,
        args.xbar,
        args.joinChar,
    )
