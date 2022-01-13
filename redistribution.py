from nltk import Tree
import argparse
import pickle

import os
import logging


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

def trees_to_cnf(trees, horzMarkov=0, vertMarkov=1):
    trees = trees.copy()
    for t in trees:
        t.chomsky_normal_form(horzMarkov=horzMarkov, vertMarkov=vertMarkov)
    return trees

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

def create_dataset_from_trees(trees):
    word_array = []
    pos_array = []
    gold_trees = []
    for tree in trees:
        token = tree.pos()
        word, pos = zip(*token)
        word_array.append(word)
        pos_array.append(pos)
        gold_trees.append(factorize(tree))

    return {'word': word_array,
            'pos': pos_array,
            'gold_tree': gold_trees}

def redistribution(args):
    # Get path for the dataset
    train_file = os.path.join(args.dir, f'{args.prefix}-train.txt')
    valid_file = os.path.join(args.dir, f'{args.prefix}-valid.txt')
    test_file = os.path.join(args.dir, f'{args.prefix}-test.txt')

    # Check save path and create if not exist
    print(f'[INFO] Dataset are saved on {args.cache_path}.')
    if not os.path.exists(args.cache_path):
        print(f'[INFO] Creating save path...', end='')
        os.makedirs(args.cache_path, exist_ok=True)
        print(f'DONE.')

    # Load dataset
    print('[INFO] Load dataset...', end='')
    train_trees = get_trees(train_file)
    valid_trees = get_trees(valid_file)
    test_trees = get_trees(test_file)
    print('DONE.')

    def depth(tree):
        return tree.height()
    def length(tree):
        return len(tree.leaves())
    def cnf_depth(tree):
        tree = tree.copy()
        tree.chomsky_normal_form()
        return tree.height()

    # Check splitting criterion
    if args.criterion != 'standard':
        split = [len(train_trees), len(train_trees) + len(valid_trees)]
        trees = [*train_trees, *valid_trees, *test_trees]

        if args.criterion == 'depth':
            criterion = depth
        elif args.criterion == 'length':
            criterion = length
        elif args.criterion == 'cnf-depth':
            criterion = cnf_depth

        print(f'Dataset distributed based on {args.criterion} of trees.')
        print(f'[INFO] Sorting...')
        trees = sorted(trees, key=lambda t : criterion(t))
        train_trees = trees[:split[0]]
        valid_trees = trees[split[0]:split[1]]
        test_trees = trees[split[1]:]

    # Print dataset status
    print(
        f'train set contain : total {len(train_trees)}\n'
        f'\tdepth: {min(map(depth, train_trees))} - {max(map(depth, train_trees))}\n'
        f'\tCNF-depth: {min(map(cnf_depth, train_trees))} - {max(map(cnf_depth, train_trees))}'
    )
    print(
        f'valid set contain : total {len(valid_trees)}\n'
        f'\tdepth: {min(map(depth, valid_trees))} - {max(map(depth, valid_trees))}\n'
        f'\tCNF-depth: {min(map(cnf_depth, valid_trees))} - {max(map(cnf_depth, valid_trees))}'
    )
    print(
        f'test set contain : total {len(test_trees)}\n'
        f'\tdepth: {min(map(depth, test_trees))} - {max(map(depth, test_trees))}\n'
        f'\tCNF-depth: {min(map(cnf_depth, test_trees))} - {max(map(cnf_depth, test_trees))}'
    )

    # Change trees to CNF trees If option is true
    if args.cnf:
        print('[INFO] Convert trees to CNF trees...', end='')
        train_trees = trees_to_cnf(train_trees)
        valid_trees = trees_to_cnf(valid_trees)
        test_trees = trees_to_cnf(test_trees)
        print('DONE.')

    print('[INFO] Saving dataset...', end='')
    result = create_dataset_from_trees(train_trees)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-train.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(valid_trees)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-valid.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(test_trees)
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
    parser.add_argument('--criterion', default='standard', choices=['standard', 'depth', 'length', 'cnf-depth'])
    parser.add_argument('--cnf', action='store_true', default=False)
    args = parser.parse_args()

    redistribution(args)