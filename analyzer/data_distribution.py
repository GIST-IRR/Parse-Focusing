#!/usr/bin/env python
import pickle
import argparse
from collections import OrderedDict, Counter, defaultdict
from numpy import vsplit
from torch.utils.tensorboard import SummaryWriter

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
import torch
from utils import span_to_tree

from parser.helper.metric import UF1


def f1(pred, gold):
    eps = 1e-8
    # in the case of sentence length=1
    if len(pred) == 0 or len(gold) == 0:
        return None
    length = max(gold, key=lambda x: x[1])[1]
    # removing the trival span
    gold = list(filter(lambda x: x[0] + 1 != x[1], gold))
    pred = list(filter(lambda x: x[0] + 1 != x[1], pred))
    # remove the entire sentence span.
    gold = list(filter(lambda x: not (x[0] == 0 and x[1] == length), gold))
    pred = list(filter(lambda x: not (x[0] == 0 and x[1] == length), pred))
    # remove label.
    gold = [g[:2] for g in gold]
    pred = [p[:2] for p in pred]
    gold = list(map(tuple, gold))
    pred = list(map(tuple, pred))

    gold = set(gold)
    pred = set(pred)
    overlap = pred.intersection(gold)
    prec = float(len(overlap)) / (len(pred) + eps)
    reca = float(len(overlap)) / (len(gold) + eps)
    if len(gold) == 0:
        reca = 1.0
        if len(pred) == 0:
            prec = 1.0
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1


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


def get_word_freq(
    dataset,
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
    threshold=10000,
):
    # Word embedding from vocab frequency
    def clean_word(words):
        import re

        def clean_number(w):
            new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
            return new_w

        return [clean_number(word.lower()) for word in words]

    new_words = []
    for w in dataset["word"]:
        w = clean_word(w)
        new_words.append(w)
    dataset["word"] = new_words

    vocab = {}
    for sent in dataset["word"]:
        for w in sent:
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1
    vocab = OrderedDict(
        sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    )
    vocab = {k: i + 2 for i, k in enumerate(vocab.keys()) if i < threshold}
    vocab.update({pad_token: 0, unk_token: 1})

    # Word number for each POS tag.
    pos_count_dict = defaultdict(lambda: [0] * len(vocab))
    for i, sent in enumerate(dataset["word"]):
        pos = dataset["pos"][i]
        for w, p in zip(sent, pos):
            if w not in vocab:
                w = vocab[unk_token]
            else:
                w = vocab[w]
            pos_count_dict[p][w] += 1
    pos_count_dict = OrderedDict(
        sorted(pos_count_dict.items(), key=lambda x: sum(x[1]))
    )
    # pos_count_dict = torch.tensor(list(pos_count_dict.values()), dtype=torch.float32)
    # pos_count_dict = pos_count_dict.log_softmax(-1)

    # torch.save(pos_count_dict, f"weights/terms.pt")
    return vocab


def get_prod_freq(dataset):
    for i, tree in enumerate(dataset["gold_tree"]):
        tree = span_to_tree(tree)
        leaf_pos = tree.treepositions("leaves")
        for j, p in enumerate(leaf_pos):
            tree[p] = dataset["word"][i][j]
        prod = tree.productions()
        print(prod)
    return prod


def get_symbols(production):
    production = production.split(" ")
    parent = production[0]
    left_child = production[2]
    if len(production) == 4:
        right_child = production[3]
    else:
        right_child = None
    return parent, left_child, right_child


def remove_double_quotes(tree):
    symbol_pos = set(tree.treepositions()) - set(tree.treepositions("leaves"))
    for i in symbol_pos:
        label = tree[i].label()
        if label == "'T'":
            tree[i].set_label("T")
        else:
            tree[i].set_label("NT")
    return tree


def parse_tree_frequency_by_length(trees):
    parse_trees = []
    lengths = defaultdict(int)
    for tree in trees:
        # Remove labels
        tree = [s[:2] for s in tree]
        # Convert to tree
        tree = span_to_tree(tree)
        # Get length of sentence
        length = len(tree.leaves())
        lengths[length] += 1

        # Remove double quotes
        tree = remove_double_quotes(tree)

        # Add to parse tree list
        if tree not in parse_trees:
            parse_trees.append(tree)

    sorted_type = defaultdict(list)
    for tree in parse_trees:
        tree_length = len(tree.leaves())
        sorted_type[tree_length].append(tree)
    print(lengths)
    return lengths


def main(args):
    """Get Analysis of Dataset.
    The analysis includes:
    - Word frequency
    - Sentence(tree) distribution for each length
    - Sentence(tree) distribution for each depth
    - Depth distribution for each length

    Args:
        dataset (_type_: str): Path to the dataset file.
        tag (_type_: str): Tag to use for TensorBoard logging.
    """
    data_name = args.dataset
    with open(data_name, "rb") as f:
        dataset = pickle.load(f)

    sort_type = "depth"

    # Save gold dataset to file
    new_dataset = {
        "word": dataset["word"],
    }

    # Compare Gold & Gold_left to get Upper bound
    # Notice that the results of gold_left is same with gold_right.
    uf1_upper = UF1()
    uf1_upper(dataset["gold_tree"], dataset["gold_tree_left"])
    prec_upper = uf1_upper.sentence_prec * 100
    reca_upper = uf1_upper.sentence_reca * 100
    sf1_upper = uf1_upper.sentence_uf1 * 100
    cf1_upper = uf1_upper.corpus_uf1 * 100
    ex_upper = uf1_upper.sentence_ex * 100

    vocab = get_word_freq(dataset)
    parse_tree_frequency_by_length(dataset["gold_tree"])
    get_prod_freq(dataset)

    # dataset transform to dict for each sentence.
    data = []
    for i in range(len(dataset["word"])):
        data.append({})

    for k, vs in dataset.items():
        for i, v in enumerate(vs):
            if k == "word":
                data[i].update({"len": len(v)})
            data[i].update({k: v})

    criterion = 40
    data = list(filter(lambda x: len(x["word"]) <= criterion, data))
    data = list(filter(lambda x: x["len"] <= criterion, data))
    data = dataset["depth"]

    # depth distribution for total data
    depth_dist = count_value(data, "len")

    # length distribution for total data
    length_dist = count_value(data, "depth")

    # depth distribution for each length
    split_dict = split_by_key(data, "len")

    n_dict = {}
    for k in split_dict.keys():
        n_dict[k] = count_value(split_dict[k], "depth")

    log_name = "log/filter_depth_dist"
    writer = SummaryWriter(log_name)

    for k, vs in n_dict.items():
        tag = f"{args.tag}/length_{k}"
        for k, v in vs.items():
            writer.add_scalar(tag, v, k)

    writer.flush()
    writer.close()

    print("=" * 30)
    print(f"Dist. for depth: {depth_dist}")
    print(f"Dist. for length: {length_dist}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    main(args)
