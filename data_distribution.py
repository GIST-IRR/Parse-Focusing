import pickle
import argparse
from collections import OrderedDict
from numpy import vsplit
from torch.utils.tensorboard import SummaryWriter

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
import torch
from utils import (
    span_to_tree
)


def split_by_key(data, key):
    # split data to ordered dictionary by key
    result = {}
    for d in data:
        v = d[key]
        if v in result:
            result[v].append(d)
        else:
            result[v] = [d]
    return OrderedDict(sorted(result.items()))

def count_value(data, key):
    result = {}
    for d in data:
        n = d[key]
        if n in result:
            result[n] += 1
        else:
            result[n] = 1
    return OrderedDict(sorted(result.items()))

def get_word_freq(dataset):
    # Word embedding from vocab frequency
    def clean_word(words):
        import re
        def clean_number(w):
            new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
            return new_w
        return [clean_number(word.lower()) for word in words]

    new_words = []
    for w in dataset['word']:
        w = clean_word(w)
        new_words.append(w)
    dataset['word'] = new_words

    vocab = {}
    for sent in dataset['word']:
        for w in sent:
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1
    vocab = OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
    vocab = {k: i+2 for i, k in enumerate(vocab.keys()) if i < 10000}
    vocab.update({'<pad>': 0, '<unk>': 1})

    pos_count_dict = {}
    for i, sent in enumerate(dataset['word']):
        pos = dataset['pos'][i]
        for w, p in zip(sent, pos):
            if w not in vocab:
                w = 1
            else:
                w = vocab[w]

            if p not in pos_count_dict:
                pos_count_dict[p] = [0] * len(vocab)
            
            pos_count_dict[p][w] += 1
    pos_count_dict = OrderedDict(
        sorted(pos_count_dict.items(), key=lambda x: sum(x[1]))
    )
    pos_count_dict = torch.tensor(
        list(pos_count_dict.values()), dtype=torch.float32
    )
    pos_count_dict = pos_count_dict.log_softmax(-1)

    torch.save(pos_count_dict, f'weights/terms.pt')

def get_prod_freq(dataset):
    for i, tree in enumerate(dataset['gold_tree']):
        tree = span_to_tree(tree)
        leaf_pos = tree.treepositions('leaves')
        for j, p in enumerate(leaf_pos):
            tree[p] = dataset['word'][i][j]
        prod = tree.productions()
        print(prod)


def get_symbols(production):
    production = production.split(' ')
    parent = production[0]
    left_child = production[2]
    if len(production) == 4:
        right_child = production[3]
    else:
        right_child = None
    return parent, left_child, right_child

def get_parse_tree_freq(trees):
    total_length = len(trees)
    parse_trees = []
    lengths = {}
    for tree in trees:
        tree = [s[:2] for s in tree]
        tree = span_to_tree(tree)
        # dist tree length
        length = len(tree.leaves())
        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1

        # dist tree type
        # symbol_pos = set(tree.treepositions()) - \
        #     set(tree.treepositions('leaves'))
        # for i in symbol_pos:
        #     label = tree[i].label()
        #     if label == "'T'":
        #         tree[i].set_label('T')
        #     else:
        #         tree[i].set_label('NT')

        # Add to parse tree list
        if tree not in parse_trees:
            parse_trees.append(tree)

    sorted_type = {}
    for tree in parse_trees:
        tree_length = len(tree.leaves())
        if tree_length not in sorted_type:
            sorted_type[tree_length] = [tree]
        else:
            sorted_type[tree_length].append(tree)
    print(lengths)


def main(args):
    data_dir = args.dir
    lang = 'edit-english'
    split = 'standard'
    form = 'both'
    data_type = f'{lang}.{split}.{form}'
    # data_type = f'{split}.{form}'
    data_split = 'train'
    data_name = f'{data_dir}/{data_type}/{lang}-{split}-{data_split}.pkl'
    # data_name = f'{data_dir}/{data_type}/chinese-{data_split}.pkl'
    with open(data_name, 'rb') as f:
        dataset = pickle.load(f)

    sort_type = 'depth'

    # get_word_freq(dataset)
    # get_parse_tree_freq(dataset['gold_tree'])
    get_prod_freq(dataset)

    # dataset transform to dict for each sentence.
    # data = []
    # for i in range(len(dataset['word'])):
    #     data.append({})

    # for k, vs in dataset.items():
    #     for i, v in enumerate(vs):
    #         if k == 'word':
    #             data[i].update({'len': len(v)})
    #         data[i].update({k: v})

    # criterion = 40
    # data = list(filter(lambda x: len(x['word']) <= criterion, data))
    # data = list(filter(lambda x: x['len'] <= criterion, data))
    # data = dataset['depth']

    # depth distribution for total data
    # depth_dist = count_value(data, 'len')

    # length distribution for total data
    # length_dist = count_value(data, 'depth')

    # depth distribution for each length
    # split_dict = split_by_key(data, 'len')
    # n_dict = {}
    # for k in split_dict.keys():
    #     n_dict[k] = count_value(split_dict[k], 'depth')

    # log_name = 'log/filter_depth_dist'
    # writer = SummaryWriter(log_name)

    # for k, vs in n_dict.items():
    #     tag = f'{data_type}/{data_split}/length_{k}'
    #     for k, v in vs.items():
    #         writer.add_scalar(tag, v, k)

    # writer.flush()
    # writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, default='data')
    parser.add_argument('--lang', default='english')
    args = parser.parse_args()

    main(args)