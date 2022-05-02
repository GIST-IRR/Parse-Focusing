from ntpath import join
from nltk import Tree
import argparse
import pickle

import os
import logging
import copy


def get_trees(file):
    trees = []
    with open(file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            trees.append(tree)
    return trees

def get_cnf_trees(file):
    trees = []
    with open(file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            tree.chomsky_normal_form(horzMarkov=0, vertMarkov=1)
            trees.append(tree)
    return trees

def tree_to_cnf(tree, factor='right', horzMarkov=None, vertMarkov=0):
    tree = tree.copy(deep=True)
    tree.chomsky_normal_form(factor=factor, horzMarkov=horzMarkov, vertMarkov=vertMarkov)
    return tree

def trees_to_cnf(trees, factor='right', horzMarkov=None, vertMarkov=0):
    result = []
    for t in trees:
        t = tree_to_cnf(t, factor=factor, horzMarkov=horzMarkov, vertMarkov=vertMarkov)
        result.append(t)
    return result

def collapse_unary(tree, collapsePOS=False, collapseRoot=False, joinChar='+'):
    tree = tree.copy(deep=True)
    tree.collapse_unary(collapsePOS=collapsePOS, collapseRoot=collapseRoot, joinChar=joinChar)
    return tree

def tree_transform(
    tree, factor='right', collapse=False, collapsePOS=True,
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
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, 'NULL']] + spans
        return j, spans
    return track(tree, 0)[1]

def create_dataset(file_name):
    word_array = []
    pos_array = []
    gold_trees = []
    with open(file_name, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            token = tree.pos()
            word, pos = zip(*token)
            word_array.append(word)
            pos_array.append(pos)
            gold_trees.append(factorize(tree))

    return {'word': word_array,
            'pos': pos_array,
            'gold_tree':gold_trees}

def create_dataset_from_trees(trees, depth=False, cnf='none', collapse=False):
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
        if cnf == 'both':
            left_tree = tree_to_cnf(col_tree, factor='left')
            gold_trees_left.append(factorize(left_tree))
            right_tree = tree_to_cnf(col_tree, factor='right')
            gold_trees_right.append(factorize(right_tree))
            if depth:
                depth_left.append(left_tree.height())
                depth_right.append(right_tree.height())
        elif cnf == 'left':
            left_tree = tree_to_cnf(col_tree, factor='left')
            gold_trees_left.append(factorize(left_tree))
            if depth:
                depth_left.append(left_tree.height())
        elif cnf == 'right':
            right_tree = tree_to_cnf(col_tree, factor='right')
            gold_trees_right.append(factorize(right_tree))
            if depth:
                depth_right.append(right_tree.height())
        if depth:
            depth_array.append(tree.height())

    result = {
        'word': word_array,
        'pos': pos_array,
        'gold_tree': gold_trees
    }
    if cnf == 'both':
        result.update({'gold_tree_left': gold_trees_left})
        result.update({'gold_tree_right': gold_trees_right})
        result.update({'depth_left': depth_left})
        result.update({'depth_right': depth_right})
    elif cnf == 'left':
        result.update({'gold_tree_left': gold_trees_left})
        result.update({'depth_left': depth_left})
    elif cnf == 'right':
        result.update({'gold_tree_right': gold_trees_right})
        result.update({'depth_right': depth_right})
    if depth:
        result.update({'depth': depth_array})

    return result

def redistribution(args):
    # Get path for the dataset
    train_file = os.path.join(args.dir, f'{args.prefix}-train.txt')
    valid_file = os.path.join(args.dir, f'{args.prefix}-valid.txt')
    test_file = os.path.join(args.dir, f'{args.prefix}-test.txt')

    # Load dataset
    print('[INFO] Load dataset...', end='')
    train_trees = get_trees(train_file)
    print('train...', end='')
    valid_trees = get_trees(valid_file)
    print('valid...', end='')
    test_trees = get_trees(test_file)
    print('test...DONE.')

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

    # Check splitting criterion
    if args.criterion != 'standard':
        if args.criterion == 'depth':
            criterion = depth
        elif args.criterion == 'length':
            criterion = length
        elif args.criterion == 'binarized-depth':
            criterion = bin_depth
        elif args.criterion == 'collapsed-depth':
            criterion = col_depth
        elif args.criterion == 'binarized-collapsed-depth':
            criterion = bin_col_depth

        split = [len(train_trees), len(train_trees) + len(valid_trees)]
        trees = [*train_trees, *valid_trees, *test_trees]

        # Sort and split
        print(f'Dataset distributed based on {args.criterion} of trees.')
        # Check sort order
        if args.reverse:
            print(f'[INFO] Sorting by reversed order...')
        else:
            print(f'[INFO] Sorting...')
        trees = sorted(trees, key=lambda t : criterion(t), reverse=args.reverse)
        train_trees = trees[:split[0]]
        valid_trees = trees[split[0]:split[1]]
        test_trees = trees[split[1]:]

    # Print dataset status
    print(
        f'train set contain : total {len(train_trees)}\n'
        f'\t{args.criterion}: {min(map(criterion, train_trees))} - {max(map(criterion, train_trees))}'
    )
    print(
        f'valid set contain : total {len(valid_trees)}\n'
        f'\t{args.criterion}: {min(map(criterion, valid_trees))} - {max(map(criterion, valid_trees))}'
    )
    print(
        f'test set contain : total {len(test_trees)}\n'
        f'\t{args.criterion}: {min(map(criterion, test_trees))} - {max(map(criterion, test_trees))}'
    )

    # Check save path and create if not exist
    print(f'[INFO] Dataset will be saved on {args.cache_path}.')
    if not os.path.exists(args.cache_path):
        print(f'[INFO] Creating save path...', end='')
        os.makedirs(args.cache_path, exist_ok=True)
        print(f'DONE.')

    # Saving datasets
    print('[INFO] Saving dataset...', end='')
    result = create_dataset_from_trees(train_trees, depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-train.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(valid_trees, depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-valid.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(test_trees, depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-test.pkl"), "wb") as f:
        pickle.dump(result, f)
    print('DONE.')

    print('Dataset distribution DONE!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--dir', required=True)
    parser.add_argument('--prefix', default='english')
    parser.add_argument('--cache_path', default='data/')
    parser.add_argument('--criterion', default='standard', choices=['standard', 'depth', 'length', 'binarized-depth', 'collapsed-depth', 'binarized-collapsed-depth'])
    parser.add_argument('--factor', default='left', choices=['left', 'right'])
    parser.add_argument('--cnf', default='none', choices=['none', 'left', 'right', 'both'])
    parser.add_argument('--target_depth', action='store_true', default=False)
    parser.add_argument('--collapse', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)
    args = parser.parse_args()

    redistribution(args)