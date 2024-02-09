#!/usr/bin/env python
import argparse
from nltk import Tree
from pathlib import Path

from utils import tree_to_span, clean_word
import torch
import pickle


def main(filepath, output, vocab):
    filepath = Path(filepath)
    output = Path(output)
    vocab = Path(vocab)

    with vocab.open("rb") as f:
        vocab = pickle.load(f)

    data = []
    with filepath.open("r") as f:
        for line in f:
            tree = Tree.fromstring(line)
            word = clean_word(tree.leaves())
            word = [vocab[w] for w in word]
            span = tree_to_span(tree)
            span = [(s[0], s[1]) for s in span]
            word_dict = {"word": word, "tree": span}
            data.append(word_dict)

    with output.open("wb") as f:
        torch.save(data, f)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--vocab", type=str)
    args = parser.parse_args()
    main(args.filepath, args.output, args.vocab)
