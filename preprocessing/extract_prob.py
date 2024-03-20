from collections import defaultdict
import argparse

import numpy as np
import pickle

from nltk import Tree
from utils import clean_word


def main(filepath, vocab, output):
    # Load trees
    trees = []
    with open(filepath, "r") as f:
        for l in f:
            trees.append(Tree.fromstring(l))

    # Load vocab
    with open(vocab, "rb") as f:
        vocab = pickle.load(f)

    # Count unary productions
    prod = defaultdict(lambda: defaultdict(int))
    for t in trees:
        pos = t.treepositions("leaves")
        unaries = [t[p[:-1]].productions()[0] for p in pos]
        for u in unaries:
            pt = u.lhs().symbol()
            w = clean_word([u.rhs()[0]])[0]
            if w not in vocab:
                w = "<unk>"
            w = vocab[w]
            prod[pt][w] += 1

    # Normalize
    prob = np.zeros((len(prod), len(vocab)))
    for i, p in enumerate(prod):
        tmp = np.zeros(len(vocab))
        total = sum(prod[p].values())
        for w in prod[p]:
            prod[p][w] /= total
            tmp[w] = prod[p][w]
        prob[i] = tmp

    # Save
    with open(output, "wb") as f:
        pickle.dump(prob, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/data.clean/edit-english-train.txt",
    )
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument("--output", type=str, default="gold_term_prob.pkl")
    args = parser.parse_args()

    main(args.filepath, args.vocab, args.output)
