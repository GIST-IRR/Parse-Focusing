#!/bin/usr/env python3
import argparse
from pathlib import Path

from nltk.tree import Tree
import numpy as np


def print_rules_ratio(
    NT_NT,
    NT_T,
    T_NT,
    T_T,
    balanced_ratio,
    unbalanced_ratio,
    total_balance_ratio,
):
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


def count_rules(trees, root=False):
    """Count the number of rules in each tree.
    And calculate the ratio of balanced and unbalanced trees.

    Args:
        trees (list of Tree): List of trees.
        root (bool, optional): It indicate that each tree have root or not. Defaults to False.

    Returns:
        tuple: Number of rules in each type and ratio of balanced and unbalanced trees.
    """
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

    return (
        NT_NT,
        NT_T,
        T_NT,
        T_T,
        balanced_ratio,
        unbalanced_ratio,
        total_balance_ratio,
    )


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

    if all(check_terminal):
        T_T += 1
    elif check_terminal[0]:
        T_NT += 1
    elif check_terminal[1]:
        NT_T += 1
    else:
        NT_NT += 1
    return NT_NT, NT_T, T_NT, T_T


def get_trees(file, factor=None):
    trees = []
    if isinstance(file, str):
        file = Path(file)
    with file.open("r") as f:
        for line in f:
            tree = Tree.fromstring(line)
            # Only consider binary trees
            if factor is not None:
                tree.chomsky_normal_form(
                    factor=factor, horzMarkov=0, vertMarkov=1
                )
                tree.collapse_unary(collapsePOS=True)
            trees.append(tree)
    return trees


def main(data_path, factor):
    data_path = Path(data_path)

    # Load dataset
    print("[INFO] Load dataset...", end="")
    trees = get_trees(data_path, factor=factor)
    print("DONE.")

    # Tree balancing
    print(f"[{data_path.stem}]")
    res = count_rules(trees)
    print_rules_ratio(*res)

    # Mean Length
    lens = [len(t.leaves()) for t in trees]
    lens = [l for l in lens if l < 40]
    mean = np.mean(lens)
    print(f"Mean length: {mean:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="data/data.clean/edit-english-train.txt"
    )
    parser.add_argument("--factor", type=str, default=None)
    args = parser.parse_args()
    main(args.dataset, args.factor)
