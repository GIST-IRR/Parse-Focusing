from nltk.tree import Tree
import argparse
import pickle
from typing import List, Union
from pathlib import Path


def get_trees(file, cnf=False):
    r"""Get parse trees from sentences in file.

    Args:
        file (str): path to file containing parse trees.
        cnf (bool): whether to convert trees to Chomsky Normal Form (CNF).

    Returns:
        list of nltk.tree.Tree: list of parse trees.

    Examples:
        >>> trees = get_trees('data/ptb/train.txt')
    """
    trees = []
    with open(file, "r") as f:
        for line in f:
            tree = Tree.fromstring(line)
            if cnf:
                tree.chomsky_normal_form(horzMarkov=0, vertMarkov=1)
            trees.append(tree)
    return trees


def trees_to_cnf(
    trees: Union[Tree, List[Tree]],
    copy=True,
    factor="right",
    horzMarkov=None,
    vertMarkov=0,
):
    r"""transform trees to Chomsky Normal Form (CNF).

    Args:
        trees (nltk.tree.Tree or list of nltk.tree.Tree): trees to be transformed.
        copy (bool): whether to copy trees before transforming.
        factor (str): factorization direction, 'right' or 'left'.
        horzMarkov (int): horizontal Markovization order.
        vertMarkov (int): vertical Markovization order.

    Returns:
        list of nltk.tree.Tree: list of transformed trees.
    """
    if isinstance(trees, Tree):
        trees = [trees]
    result = []
    for t in trees:
        if copy:
            t = t.copy(deep=True)
        t.chomsky_normal_form(
            factor=factor, horzMarkov=horzMarkov, vertMarkov=vertMarkov
        )
        result.append(t)
    return result


def collapse_unary(tree, collapsePOS=False, collapseRoot=False, joinChar="+"):
    tree = tree.copy(deep=True)
    tree.collapse_unary(
        collapsePOS=collapsePOS, collapseRoot=collapseRoot, joinChar=joinChar
    )
    return tree


def tree_transform(
    tree,
    factor="right",
    collapse=False,
    collapsePOS=True,
):
    tree = tree.copy(deep=True)
    if collapse:
        tree.collapse_unary(collapsePOS=collapsePOS)
    tree.chomsky_normal_form(factor=factor)
    return tree


def factorize(tree):
    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i + 1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, "NULL"]] + spans
        return j, spans

    return track(tree, 0)[1]


def create_dataset_from_trees(trees, depth=False, cnf="none", collapse=False):
    word_array = []
    pos_array = []
    gold_trees = []
    gold_trees_left = []
    gold_trees_right = []
    depth_array = []
    depth_left = []
    depth_right = []
    for tree in trees:
        if collapse:
            col_tree = collapse_unary(tree, collapsePOS=True)
        else:
            col_tree = tree
        token = tree.pos()
        word, pos = zip(*token)
        word_array.append(word)
        pos_array.append(pos)
        gold_trees.append(factorize(tree))
        if cnf == "both":
            left_tree = trees_to_cnf(col_tree, factor="left")[0]
            gold_trees_left.append(factorize(left_tree))
            right_tree = trees_to_cnf(col_tree, factor="right")[0]
            gold_trees_right.append(factorize(right_tree))
            if depth:
                depth_left.append(left_tree.height())
                depth_right.append(right_tree.height())
        elif cnf == "left":
            left_tree = trees_to_cnf(col_tree, factor="left")[0]
            gold_trees_left.append(factorize(left_tree))
            if depth:
                depth_left.append(left_tree.height())
        elif cnf == "right":
            right_tree = trees_to_cnf(col_tree, factor="right")[0]
            gold_trees_right.append(factorize(right_tree))
            if depth:
                depth_right.append(right_tree.height())
        if depth:
            depth_array.append(tree.height())

    result = {"word": word_array, "pos": pos_array, "gold_tree": gold_trees}
    if cnf == "both":
        result.update({"gold_tree_left": gold_trees_left})
        result.update({"gold_tree_right": gold_trees_right})
        result.update({"depth_left": depth_left})
        result.update({"depth_right": depth_right})
    elif cnf == "left":
        result.update({"gold_tree_left": gold_trees_left})
        result.update({"depth_left": depth_left})
    elif cnf == "right":
        result.update({"gold_tree_right": gold_trees_right})
        result.update({"depth_right": depth_right})
    if depth:
        result.update({"depth": depth_array})

    return result


def redistribution(args):
    # Get path for the dataset
    train_file = args.dir / f"{args.prefix}-train.txt"
    valid_file = args.dir / f"{args.prefix}-valid.txt"
    test_file = args.dir / f"{args.prefix}-test.txt"

    # Load dataset
    print("[INFO] Load dataset...", end="")
    if not args.no_train:
        print("train...", end="")
        train_trees = get_trees(train_file)
    else:
        train_trees = []
    if not args.no_valid:
        print("valid...", end="")
        valid_trees = get_trees(valid_file)
    else:
        valid_trees = []
    if not args.no_test:
        print("test...", end="")
        test_trees = get_trees(test_file)
    else:
        test_trees = []
    print("DONE.")

    # Define criterion for splitting
    def depth(tree):
        return tree.height()

    def length(tree):
        return len(tree.leaves())

    def bin_depth(tree):
        t = tree.copy(deep=True)
        t.chomsky_normal_form(factor=args.factor)
        return t.height()

    def col_depth(tree):
        t = tree.copy(deep=True)
        t.collapse_unary()
        return t.height()

    def bin_col_depth(tree):
        t = tree.copy(deep=True)
        t.collapse_unary()
        t.chomsky_normal_form(factor=args.factor)
        return t.height()

    # Status Test
    print("[INFO] Status Test")
    for trees, tag in zip(
        [train_trees, valid_trees, test_trees], ["train", "valid", "test"]
    ):
        if len(trees) != 0:
            print(
                f"{tag} set contain : total {len(trees)}\n"
                f"\tLength: {min(map(length, trees))} - {max(map(length, trees))}\n"
                f"\tDepth: {min(map(depth, trees))} - {max(map(depth, trees))}"
            )

    # Check splitting criterion
    if args.criterion != "standard":
        if args.criterion == "depth":
            criterion = depth
        elif args.criterion == "length":
            criterion = length
        elif args.criterion == "binarized-depth":
            criterion = bin_depth
        elif args.criterion == "collapsed-depth":
            criterion = col_depth
        elif args.criterion == "binarized-collapsed-depth":
            criterion = bin_col_depth

        split = [len(train_trees), len(train_trees) + len(valid_trees)]
        trees = [*train_trees, *valid_trees, *test_trees]

        # Sort and split
        print(f"Dataset distributed based on {args.criterion} of trees.")
        # Check sort order
        if args.reverse:
            print(f"[INFO] Sorting by reversed order...")
        else:
            print(f"[INFO] Sorting...")
        trees = sorted(trees, key=lambda t: criterion(t), reverse=args.reverse)
        train_trees = trees[: split[0]]
        valid_trees = trees[split[0] : split[1]]
        test_trees = trees[split[1] :]

        # Print dataset status
        for trees in [train_trees, valid_trees, test_trees]:
            print(
                f"train set contain : total {len(trees)}\n"
                f"\t{args.criterion}: {min(map(criterion, trees))} - {max(map(criterion, trees))}"
            )

    # Check save path and create if not exist
    print(f"[INFO] Dataset will be saved on {args.save_dir}.")
    if not args.save_dir.exists():
        print(f"[INFO] Creating save path...", end="")
        args.save_dir.mkdir(exist_ok=True)
        print(f"DONE.")

    # # save sentence
    # def save_sentences(trees, tag):
    #     sent = [" ".join(t.leaves()) + "\n" for t in trees]
    #     with (args.save_dir / f"{tag}.txt").open("w") as f:
    #         for s in sent:
    #             f.write(s)

    # save_sentences(train_trees, "train")
    # save_sentences(valid_trees, "valid")
    # save_sentences(test_trees, "test")
    # exit()

    # Saving datasets
    def save_trees(trees, tag):
        tree_size = len(trees)
        result = create_dataset_from_trees(
            trees[:tree_size],
            depth=args.target_depth,
            cnf=args.cnf,
            collapse=args.collapse,
        )
        path = args.save_dir / f"{args.prefix}-{args.criterion}-{tag}.pkl"
        with path.open("wb") as f:
            pickle.dump(result, f)

    print("[INFO] Saving dataset...", end="")
    if not args.no_train:
        print("train...", end="")
        save_trees(train_trees, "train")
    if not args.no_valid:
        print("valid...", end="")
        save_trees(valid_trees, "valid")
    if not args.no_test:
        print("test...", end="")
        save_trees(test_trees, "test")
    print("DONE.")

    print("Dataset distribution DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess ptb file.")
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--no-train", default=False, action="store_true")
    parser.add_argument("--no-valid", default=False, action="store_true")
    parser.add_argument("--no-test", default=False, action="store_true")
    parser.add_argument("--prefix", default="english")
    parser.add_argument("--save_dir", default=Path("data/"), type=Path)
    parser.add_argument(
        "--criterion",
        default="standard",
        choices=[
            "standard",
            "depth",
            "length",
            "binarized-depth",
            "collapsed-depth",
            "binarized-collapsed-depth",
        ],
    )
    parser.add_argument("--factor", default="left", choices=["left", "right"])
    parser.add_argument(
        "--cnf", default="none", choices=["none", "left", "right", "both"]
    )
    parser.add_argument("--target_depth", action="store_true", default=False)
    parser.add_argument("--collapse", action="store_true", default=False)
    parser.add_argument("--reverse", action="store_true", default=False)
    args = parser.parse_args()

    redistribution(args)
