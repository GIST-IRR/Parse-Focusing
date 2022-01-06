from nltk import Tree
import argparse
import pickle

import os


def get_trees(file):
    trees = []
    with open(file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            trees.append(tree)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--dir', required=True)
    parser.add_argument('--prefix', default='ptb')
    parser.add_argument('--cache_path', default='data/')
    parser.add_argument('--criterion', default='depth', choices=['depth', 'length'])
    args = parser.parse_args()

    train_file = os.path.join(args.dir, f'{args.prefix}-train.txt')
    valid_file = os.path.join(args.dir, f'{args.prefix}-valid.txt')
    test_file = os.path.join(args.dir, f'{args.prefix}-test.txt')

    print('[INFO] Load dataset...', end='')
    train_trees = get_trees(train_file)
    valid_trees = get_trees(valid_file)
    test_trees = get_trees(test_file)
    print('DONE.')

    split = [len(train_trees), len(train_trees) + len(valid_trees)]
    trees = [*train_trees, *valid_trees, *test_trees]

    def depth(tree):
        return tree.height()
    def length(tree):
        return len(tree.leaves())

    if args.criterion == 'depth':
        criterion = depth
    elif args.criterion == 'length':
        criterion = length

    print(f'[INFO] Dataset distributed based on {args.criterion} of trees.')
    print(f'[INFO] Sorting...')
    trees = sorted(trees, key=lambda t : criterion(t))
    train_trees = trees[:split[0]]
    valid_trees = trees[split[0]:split[1]]
    test_trees = trees[split[1]:]

    print(f'[INFO] train set contain {args.criterion} {criterion(train_trees[0])} from {args.criterion} {criterion(train_trees[-1])}: total {len(train_trees)}')
    print(f'[INFO] valid set contain {args.criterion} {criterion(valid_trees[0])} from {args.criterion} {criterion(valid_trees[-1])}: total {len(valid_trees)}')
    print(f'[INFO] test set contain {args.criterion} {criterion(test_trees[0])} from {args.criterion} {criterion(test_trees[-1])}: total {len(test_trees)}')

    print(f'[INFO] Dataset are saved on {args.cache_path}.')
    if not os.path.exists(args.cache_path):
        print(f'[INFO] Creating save path...', end='')
        os.makedirs(args.cache_path, exist_ok=True)
        print(f'DONE.')

    result = create_dataset_from_trees(train_trees)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-train.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(valid_trees)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-valid.pkl"), "wb") as f:
        pickle.dump(result, f)

    result = create_dataset_from_trees(test_trees)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-test.pkl"), "wb") as f:
        pickle.dump(result, f)

    print('Dataset distribution DONE!')