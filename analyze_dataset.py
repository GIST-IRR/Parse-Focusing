import os
import pickle
from pathlib import Path

from nltk.tree import Tree
from easydict import EasyDict as edict
import numpy as np

from utils import span_to_tree


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

    print(f'NT_NT: {NT_NT} ({NT_NT_ratio:.2%})')
    print(f'NT_T: {NT_T} ({NT_T_ratio:.2%})')
    print(f'T_NT: {T_NT} ({T_NT_ratio:.2%})')
    print(f'T_T: {T_T} ({T_T_ratio:.2%})')
    print(f'balanced: {balanced_ratio:.2%}')
    print(f'unbalanced: {unbalanced_ratio:.2%}')
    print(f'total_balance: {total_balance_ratio:.2%}')

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
    with open(file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            if factor is not None:
                tree.chomsky_normal_form(
                    factor=factor, horzMarkov=0, vertMarkov=1
                )
                tree.collapse_unary(collapsePOS=True)
            trees.append(tree)
    return trees

# Get path for the dataset
args = {
    'dir': 'data/data.clean',
    'prefix': 'korean',
    'postfix': 'valid',
}
args = edict(args)

langs = [
    'edit-english',
    'edit-chinese',
    'basque',
    'french',
    'german',
    'hebrew',
    'hungarian',
    'korean',
    'polish',
    'swedish',
]

# Binary rules parse tree
# for lang in langs:
#     args.prefix = lang
#     data_file = os.path.join(args.dir, f'{args.prefix}-{args.postfix}.txt')
#     # valid_file = os.path.join(args.dir, f'{args.prefix}-valid.txt')
#     # test_file = os.path.join(args.dir, f'{args.prefix}-test.txt')

#     # Load dataset
#     print('[INFO] Load dataset...', end='')
#     # trees_left = get_trees(data_file, factor='left')
#     # trees_right = get_trees(data_file, factor='right')
#     trees = get_trees(data_file)
#     # print('train...', end='')
#     # valid_trees = get_trees(valid_file)
#     # print('valid...', end='')
#     # test_trees = get_trees(test_file)
#     # print('test...', end='')
#     print('DONE.')

#     # print(f'[{args.prefix}-{args.postfix}-left]')
#     # count_rules(trees_left)

#     # print(f'[{args.prefix}-{args.postfix}-right]')
#     # count_rules(trees_right)
    
#     print(f'[{args.prefix}-{args.postfix}]')
#     count_rules(trees)

# Non-binary rules parse trees
# for lang in langs:
#     args.prefix = lang
#     data_file = os.path.join(args.dir, f'{args.prefix}-{args.postfix}.txt')

#     # Load dataset
#     trees = get_trees(data_file)
#     lens = [len(t.leaves()) for t in trees]
#     lens = [l for l in lens if l < 40]
#     mean = np.mean(lens)
    
#     print(f'[{args.prefix}-{args.postfix}]')
#     print(f'mean: {mean:.2f}')

# Predicted parse trees
dir_path = sorted(Path('log').glob('n_*_std_orig_long_seed'))
file_name = 'parse_tree.pickle'
for parent_path in dir_path:
    if parent_path.name == 'n_english_std_orig_seed':
        continue
    print(f'[{parent_path}]')
    for child_path in parent_path.iterdir():
        if not child_path.is_dir():
            continue
        print(f'[{child_path.name}]')
        with open(child_path / file_name, 'rb') as f:
            trees = pickle.load(f)
        trees = trees['trees']
        pred_trees = [span_to_tree(t['pred_tree']) for t in trees]
        count_rules(pred_trees, root=False)