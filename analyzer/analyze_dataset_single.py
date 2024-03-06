#!/bin/usr/env python3
"""
Name: analyze_dataset.py
Description: Analyze ratio for type of binary rules on the gold parse trees.
binary rules are composed with four types of rules(NT_NT, NT_T, T_NT, T_T).
In case that the ratio of NT_NT and T_T is higher than NT_T and T_NT,
the tree is called balanced tree.
On the other hands, the tree is called unbalanced tree.
"""
from pathlib import Path
from collections import defaultdict

from nltk.tree import Tree
import numpy as np


def count_rules(trees, root=False):
    NT_NT = 0
    NT_T = 0
    T_NT = 0
    T_T = 0

    balanced_ratio = 0
    unbalanced_ratio = 0

    total_balance_ratio = 0

    for tree in trees:
        if root:
            tree = tree[0]
        nt_nt, nt_t, t_nt, t_t = check_rules(tree)

        total = nt_nt + nt_t + t_nt + t_t
        br = (nt_nt + t_t) / total
        ur = (nt_t + t_nt) / total
        balanced_ratio += br
        unbalanced_ratio += ur

        total_balance_ratio += 1 if br > ur else 0

        NT_NT += nt_nt
        NT_T += nt_t
        T_NT += t_nt
        T_T += t_t

    balanced_ratio /= len(trees)
    unbalanced_ratio /= len(trees)
    total_balance_ratio /= len(trees)

    NT_NT_ratio = NT_NT / (NT_NT + NT_T + T_NT + T_T)
    NT_T_ratio = NT_T / (NT_NT + NT_T + T_NT + T_T)
    T_NT_ratio = T_NT / (NT_NT + NT_T + T_NT + T_T)
    T_T_ratio = T_T / (NT_NT + NT_T + T_NT + T_T)

    print(f"NT_NT: {NT_NT} ({NT_NT_ratio:.2%})")
    print(f"NT_T: {NT_T} ({NT_T_ratio:.2%})")
    print(f"T_NT: {T_NT} ({T_NT_ratio:.2%})")
    print(f"T_T: {T_T} ({T_T_ratio:.2%})")
    print(f"balanced: {balanced_ratio:.2%}")
    print(f"unbalanced: {unbalanced_ratio:.2%}")
    print(f"total_balance: {total_balance_ratio:.2%}")

    return NT_NT, NT_T, T_NT, T_T


def check_rules(tree):
    check_terminal = [False, False]
    NT_NT = 0
    NT_T = 0
    T_NT = 0
    T_T = 0

    for i, c in enumerate(tree):
        if len(c) == 1:
            check_terminal[i] = True
        elif len(c) == 2:
            nt_nt, nt_t, t_nt, t_t = check_rules(c)
            NT_NT += nt_nt
            NT_T += nt_t
            T_NT += t_nt
            T_T += t_t
    if check_terminal[0] and check_terminal[1]:
        T_T += 1
    elif check_terminal[0]:
        T_NT += 1
    elif check_terminal[1]:
        NT_T += 1
    else:
        NT_NT += 1
    return NT_NT, NT_T, T_NT, T_T


def check_rules_nonbinary(tree):
    check_terminal = [False] * len(tree)
    NT_NT = 0
    NT_T = 0
    T_NT = 0
    T_T = 0

    for i, c in enumerate(tree):
        if isinstance(c[0], str):
            check_terminal[i] = True
        elif len(c) == 2:
            nt_nt, nt_t, t_nt, t_t = check_rules(c)
            NT_NT += nt_nt
            NT_T += nt_t
            T_NT += t_nt
            T_T += t_t
    if check_terminal[0] and check_terminal[1]:
        T_T += 1
    elif check_terminal[0]:
        T_NT += 1
    elif check_terminal[1]:
        NT_T += 1
    else:
        NT_NT += 1
    return NT_NT, NT_T, T_NT, T_T


# get tree and count types of rules
def get_tree(tree):
    if isinstance(tree, str):
        return tree
    else:
        return [get_tree(t) for t in tree]


def get_trees(file, factor=None):
    trees = []
    if isinstance(file, str):
        file = Path(file)
    with file.open("r") as f:
        for line in f:
            tree = Tree.fromstring(line)
            if factor is not None:
                tree.chomsky_normal_form(
                    factor=factor, horzMarkov=0, vertMarkov=1
                )
                tree.collapse_unary(collapsePOS=True)
            trees.append(tree)
    return trees


def get_pos(tree, mode="both"):
    pos = tree.treepositions("leaves")
    if mode == "both":
        return [(tree[p[:-1]].label(), tree[p]) for p in pos]
    elif mode == "pos":
        return [tree[p[:-1]].label() for p in pos]
    elif mode == "leaves":
        return [tree[p] for p in pos]
    # Counter(list(filter(lambda x: x[1] == "buch", pos_n_word)))


def main(args):
    data_path = Path(args.dataset)

    # Load dataset
    print("[INFO] Load dataset...", end="")
    trees = get_trees(data_path, factor=args.factor)
    print("DONE.")

    # Tree balancing
    print(f"[{data_path.stem}]")
    count_rules(trees)

    # Mean Length
    lens = [len(t.leaves()) for t in trees]
    lens = [l for l in lens if l < 40]
    mean = np.mean(lens)
    print(f"Mean length: {mean:.2f}")

    # Vocabulary
    thr = 60
    vocab = defaultdict(int)
    for t in trees:
        for w in t.leaves():
            vocab[w] += 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab_sum = sum([e[1] for e in vocab])

    def unknown_masking(vocab, thr):
        unk_count = 0
        for i, e in enumerate(vocab):
            if i >= thr:
                unk_count += e[1]
        vocab = vocab[:thr]
        vocab.insert(0, ("<unk>", unk_count))
        return vocab

    vocab = unknown_masking(vocab, thr)
    vocab_ratio = [(e[0], e[1] / vocab_sum) for e in vocab]

    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="data/data.clean/edit-english-train.txt"
    )
    parser.add_argument("--factor", type=str, default=None)
    args = parser.parse_args()
    main(args)
