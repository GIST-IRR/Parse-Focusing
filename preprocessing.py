from nltk.tree import Tree
from nltk.grammar import Nonterminal

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary

import argparse
import pickle
from collections import defaultdict

import os
from typing import List, Union

import torch


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
    with open(file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            if cnf:
                tree.chomsky_normal_form(horzMarkov=0, vertMarkov=1)
            trees.append(tree)
    return trees

def trees_to_cnf(
    trees: Union[Tree, List[Tree]],
    copy=True, factor='right', horzMarkov=None, vertMarkov=0
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
            left_tree = trees_to_cnf(col_tree, factor='left')
            gold_trees_left.append(factorize(left_tree))
            right_tree = trees_to_cnf(col_tree, factor='right')
            gold_trees_right.append(factorize(right_tree))
            if depth:
                depth_left.append(left_tree.height())
                depth_right.append(right_tree.height())
        elif cnf == 'left':
            left_tree = trees_to_cnf(col_tree, factor='left')
            gold_trees_left.append(factorize(left_tree))
            if depth:
                depth_left.append(left_tree.height())
        elif cnf == 'right':
            right_tree = trees_to_cnf(col_tree, factor='right')
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

def create_probability_distribution(
    dir, prefix, postfix, collapsePos=True, factor='right'
    ):
    file = os.path.join(dir, f'{prefix}-{postfix}.txt')

    print('[INFO] Load dataset...', end='')
    trees = get_trees(file)
    print('Done')

    # Define criterion for splitting
    def depth(tree):
        return tree.height()
    def length(tree):
        return len(tree.leaves())

    print('[INFO] Status Test')
    print(
        f'{postfix} set contain : total {len(trees)}\n'
        f'\tLength: {min(map(length, trees))} - {max(map(length, trees))}\n'
        f'\tDepth: {min(map(depth, trees))} - {max(map(depth, trees))}'
    )

    root_productions = defaultdict(int)
    rule_productions = defaultdict(int)
    term_productions = defaultdict(int)

    def clean_word(words):
        import re
        def clean_number(w):
            new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
            return new_w
        return [clean_number(word.lower()) for word in words]

    def remove_indicator(p, indicator='*'):
        p._lhs._symbol = p.lhs().symbol().split(indicator)[-1]
        p._lhs._symbol = p.lhs().symbol().split('|')[0]

        if isinstance(p.rhs()[0], Nonterminal):
            p._rhs[0]._symbol = p.rhs()[0].symbol().split(indicator)[-1]
            p._rhs[0]._symbol = p.rhs()[0].symbol().split('|')[0]
            pass
        else:
            p._rhs = (clean_word([p.rhs()[0]])[0], )

        if len(p.rhs()) == 2:
            p._rhs[1]._symbol = p.rhs()[1].symbol().split(indicator)[-1]
            p._rhs[1]._symbol = p.rhs()[1].symbol().split('|')[0]
            pass

        return p
    
    words = [t.leaves() for t in trees]
    dataset = DataSet()
    dataset.add_field('word', words)
    dataset.apply_field(clean_word, 'word', 'word')
    vocab = Vocabulary(max_size=10000)
    vocab.from_dataset(dataset, field_name='word')

    for tree in trees:
        tree.chomsky_normal_form(factor=factor, horzMarkov=0, vertMarkov=0)
        tree.collapse_unary(collapsePOS=collapsePos, joinChar='*')
        prods = tree.productions()
        for p in prods:
            p = remove_indicator(p)
            # lhs = p.lhs()

            # Check whether the production is a unary or binary
            if len(p.rhs()) == 1:
                rhs = p.rhs()[0]
                if isinstance(rhs, str):
                    # Unary Rules
                    term_productions[p] += 1
                else:
                    # ROOT -> S
                    root_productions[p] += 1
            elif len(p.rhs()) == 2:
                rule_productions[p] += 1
    
    # Count symbols
    nonterminals = defaultdict(int)
    terminals = defaultdict(int)
    # vocabulary = defaultdict(int)

    for p in term_productions:
        lhs = p.lhs()
        rhs = p.rhs()[0]
        terminals[lhs] += 1
        # vocabulary[rhs] += 1

    for p in rule_productions:
        lhs = p.lhs()
        rhs = p.rhs()
        if lhs in terminals:
            lhs._symbol = 'NT-' + lhs.symbol()
        nonterminals[lhs] += 1
        if rhs[0] not in terminals:
            nonterminals[rhs[0]] += 1
        else:
            terminals[rhs[0]] += 1
        if rhs[1] not in terminals:
            nonterminals[rhs[1]] += 1
        else:
            terminals[rhs[1]] += 1

    for p in root_productions:
        rhs = p.rhs()[0]
        if rhs not in terminals:
            nonterminals[rhs] += 1
    
    # Construct the probability distribution
    nonterminals = list(dict(sorted(
        nonterminals.items(), key=lambda x: x[1], reverse=True
    )).keys())
    terminals = list(dict(sorted(
        terminals.items(), key=lambda x: x[1], reverse=True
    )).keys())
    # vocabulary = list(dict(sorted(
    #     vocabulary.items(), key=lambda x: x[1], reverse=True
    # )).keys())

    symbols = {s: i for i, s in enumerate(nonterminals)}
    num_nt = len(nonterminals)
    num_t = len(terminals)
    terminals = {s: i+num_nt for i, s in enumerate(terminals)}
    symbols.update(terminals)
    # tokens = {'<pad>': 0, '<unk>': 1}
    # tokens.update({s: i+2 for i, s in enumerate(vocabulary) if i < 10000})
    print(f'[INFO] Nonterminal size: {num_nt}')
    print(f'[INFO] Terminal size: {num_t}')

    # Symbol productions to index productions
    root_pd = [[0] * num_nt]
    for p, c in root_productions.items():
        rhs_idx = symbols[p.rhs()[0]]
        if rhs_idx >= num_nt:
            continue
        root_pd[0][rhs_idx] = c

    rule_pd = [
        [[0] * len(symbols) for _ in range(len(symbols))]
        for _ in range(len(nonterminals))
    ]
    for p, c in rule_productions.items():
        lhs_idx = symbols[p.lhs()]
        rhs_l_idx = symbols[p.rhs()[0]]
        rhs_r_idx = symbols[p.rhs()[1]]
        rule_pd[lhs_idx][rhs_l_idx][rhs_r_idx] = c

    term_pd = [[0] * len(vocab) for _ in range(num_t)]
    for p, c in term_productions.items():
        lhs_idx = symbols[p.lhs()] - num_nt
        rhs_idx = vocab.word2idx.get(p.rhs()[0], 1)
        term_pd[lhs_idx][rhs_idx] += c

    root_pd = torch.tensor(root_pd).float()
    rule_pd = torch.tensor(rule_pd).float()
    term_pd = torch.tensor(term_pd).float()

    torch.save(root_pd, f'weights/{prefix}_xbar_{factor}_root_pd.pt')
    torch.save(rule_pd, f'weights/{prefix}_xbar_{factor}_rule_pd.pt')
    torch.save(term_pd, f'weights/{prefix}_xbar_{factor}_term_pd.pt')
    print("Done.")

def redistribution(args):
    create_probability_distribution(
        args.dir, args.prefix, 'train', factor=args.factor)
    exit()
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

    # Status Test
    print('[INFO] Status Test')
    for trees in [train_trees, valid_trees, test_trees]:
        print(
            f'train set contain : total {len(trees)}\n'
            f'\tLength: {min(map(length, trees))} - {max(map(length, trees))}\n'
            f'\tDepth: {min(map(depth, trees))} - {max(map(depth, trees))}'
        )

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
        for trees in [train_trees, valid_trees, test_trees]:
            print(
                f'train set contain : total {len(trees)}\n'
                f'\t{args.criterion}: {min(map(criterion, trees))} - {max(map(criterion, trees))}'
            )

    # Check save path and create if not exist
    print(f'[INFO] Dataset will be saved on {args.cache_path}.')
    if not os.path.exists(args.cache_path):
        print(f'[INFO] Creating save path...', end='')
        os.makedirs(args.cache_path, exist_ok=True)
        print(f'DONE.')

    # Saving datasets
    print('[INFO] Saving dataset...', end='')
    tree_size = len(train_trees) // 8
    result = create_dataset_from_trees(train_trees[:tree_size], depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-train.pkl"), "wb") as f:
        pickle.dump(result, f)

    tree_size = len(valid_trees) // 8
    result = create_dataset_from_trees(valid_trees[:tree_size], depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-valid.pkl"), "wb") as f:
        pickle.dump(result, f)

    tree_size = len(test_trees) // 8
    result = create_dataset_from_trees(test_trees[:tree_size], depth=args.target_depth, cnf=args.cnf, collapse=args.collapse)
    with open(os.path.join(args.cache_path, f"{args.prefix}-{args.criterion}-test.pkl"), "wb") as f:
        pickle.dump(result, f)
    print('DONE.')

    print('Dataset distribution DONE!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--dir', required=True)
    parser.add_argument('--train', default=False)
    parser.add_argument('--valid', default=False)
    parser.add_argument('--test', default=True)
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