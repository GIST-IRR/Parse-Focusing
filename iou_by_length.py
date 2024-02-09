import pickle
import torch
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from utils import span_to_list, span_to_tree, sort_span
import itertools

import numpy as np


def open_trees(filename):
    filepath = Path(filename)
    with filepath.open("rb") as f:
        if filepath.suffix == ".pkl":
            trees = pickle.load(f)
        elif filepath.suffix == ".pt":
            trees = torch.load(f)
    return trees


def open_pickle(filename):
    with Path(filename).open("rb") as f:
        obj = pickle.load(f)
    return obj


def preprocess_span(span, length):
    span = list(filter(lambda x: x[0] + 1 != x[1], span))
    span = list(filter(lambda x: not (x[0] == 0 and x[1] == length), span))
    span = [g[:2] for g in span]
    span = list(map(tuple, span))
    span = set(span)
    return span


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


# def IoU(pred, gold):
#     # in the case of sentence length=1
#     if len(pred) == 0 or len(gold) == 0:
#         return None
#     length = max(gold,key=lambda x:x[1])[1]

#     gold = preprocess_span(gold, length)
#     pred = preprocess_span(pred, length)

#     union = pred.union(gold)
#     intersection = pred.intersection(gold)

#     if len(union) == 0:
#         return 0

#     res = len(intersection) / len(union)
#     return res


def IoU(*preds):
    for p in preds:
        if len(p) == 0:
            return None
    length = max(preds[0], key=lambda x: x[1])[1]
    preds = [preprocess_span(pred, length) for pred in preds]

    union = set.union(*preds)
    intersection = set.intersection(*preds)

    if len(union) == 0:
        return 0

    res = len(intersection) / len(union)
    return res


def dice(pred, gold):
    # in the case of sentence length=1
    if len(pred) == 0:
        return None
    length = max(gold, key=lambda x: x[1])[1]

    gold = preprocess_span(gold, length)
    pred = preprocess_span(pred, length)

    union = pred.union(gold)
    intersection = pred.intersection(gold)

    if len(union) == 0:
        return 0

    res = 2 * len(intersection) / len(union) + len(intersection)
    return res


def intersection(preds):
    # in the case of sentence length=1
    if len(preds) == 0:
        return None
    length = max(preds[0], key=lambda x: x[1])[1]

    preds = [preprocess_span(pred, length) for pred in preds]
    intersection = set.intersection(*preds)

    return intersection


def set_leaves(tree, words):
    tp = tree.treepositions("leaves")
    for i, p in enumerate(tp):
        tree[p] = str(words[i])


def pretty_print(tree):
    word = tree["word"]
    gold = span_to_tree(tree["gold"])
    pred1 = span_to_tree(tree["pred1"])
    pred2 = span_to_tree(tree["pred2"])

    set_leaves(gold, word)
    set_leaves(pred1, word)
    set_leaves(pred2, word)

    gold.pretty_print()
    pred1.pretty_print()
    pred2.pretty_print()


def main(args):
    if args.vocab is not None:
        vocab = open_trees(args.vocab)
        vocab = vocab["vocab"]
    else:
        vocab = None

    trees_list = args.trees
    trees = []
    for t in trees_list:
        t = open_trees(t)
        if isinstance(t, dict):
            if vocab is None:
                vocab = t["vocab"]
            t = t["trees"]
        elif isinstance(t, list):
            t = t
        t = list(filter(lambda x: len(x["sentence"]) > 1, t))
        t = sorted(t, key=lambda x: (len(x["word"]), x["word"], x["sentence"]))
        if not "pred_tree" in t[0].keys():
            for e in t:
                e["pred_tree"] = e["tree"]
                del e["tree"]
        trees.append(t)

    def decode(tree):
        word_idx = tree["word"]
        word = []
        for w in word_idx:
            word.append(vocab.to_word(w))
        tree["word"] = word

    for tree in trees:
        for t in tree:
            decode(t)

    filtered_tree = []
    totally_same = []
    different = []
    diff_n_bigger = []
    f1s = [[] for _ in range(len(trees))]
    comb = list(itertools.combinations(range(len(trees)), 2))
    iou_bts = [[] for _ in range(len(comb))]
    n_spans = 0
    shared_span = 0
    iou_tots = []

    pred_same_struct = defaultdict(list)

    gold_frequencies = Counter()
    left_gold_frequencies = Counter()
    right_gold_frequencies = Counter()
    frequencies = [Counter() for _ in range(len(trees))]

    maximum_trees = []
    minimum_trees = []

    correct_fixed = 0
    wrong_fixed = 0
    num_fluct = []

    for ts in zip(*trees):
        word = ts[0]["word"]
        # if len(word) <= 2:
        #     continue
        gold = ts[0]["gold_tree"]

        pred = [[s[:2] for s in t["pred_tree"]] for t in ts]
        for p in pred:
            if len(p) < len(word):
                p.append((0, len(word)))
                for i in range(len(word)):
                    p.append((i, i + 1))
        pred = sort_span(pred)

        scores = [f1(p, gold) for p in pred]

        iou = [IoU(p, gold) for p in pred]
        iou_bt = [IoU(pred[e[0]], pred[e[1]]) for e in comb]
        iou_tot = IoU(*pred)

        intersect = intersection(pred)
        p_gold = preprocess_span(gold, len(word))
        for i in intersect:
            if i in p_gold:
                correct_fixed += 1
            else:
                wrong_fixed += 1
        fluct = len(preprocess_span(pred[0], len(word)))
        num_fluct.append(fluct)

        for f, s in zip(f1s, scores):
            f.append(s)

        for ib, s in zip(iou_bts, iou_bt):
            ib.append(s)

        iou_tots.append(iou_tot)

        item = {
            "word": word,
            "gold": gold,
            "mean": np.mean(scores),
            "var": np.var(scores),
        }

        for i, p in enumerate(pred):
            # item[f'pred{i+1}'] = preprocess_span(p, len(word))
            item[f"pred{i+1}"] = p
            item[f"score{i+1}"] = scores[i]
            item[f"iou{i+1}"] = iou[i]
            item[f"iou_bt{i+1}"] = iou_bt[i]

        if item["var"] > 0.0:
            filtered_tree.append(item)

        if all(pred[0] == p for p in pred):
            totally_same.append(item)
        else:
            different.append(item)

        # if len(word) > 2 and scores[0] > scores[1] and iou_bt[0] < 0.5:
        if len(word) > 2 and scores[0] > scores[1]:
            diff_n_bigger.append(item)

        if len(word) > 2:
            k = frozenset([tuple(e[:2]) for e in gold])
            pred_same_struct[k].append(item)

        # Count rules frequencies
        gold_no_label = [tuple(e[:2]) for e in gold]
        gold_tree = span_to_tree(gold_no_label)
        left_gold_tree = gold_tree.copy()
        left_gold_tree.chomsky_normal_form("left")
        right_gold_tree = gold_tree.copy()
        right_gold_tree.chomsky_normal_form("right")

        f_gold = Counter(gold_tree.productions())
        f_left_gold = Counter(left_gold_tree.productions())
        f_right_gold = Counter(right_gold_tree.productions())
        gold_frequencies += f_gold
        left_gold_frequencies += f_left_gold
        right_gold_frequencies += f_right_gold
        for i, p in enumerate(pred):
            f_r = Counter(span_to_tree(p).productions())
            frequencies[i] += f_r

        # Maximum trees
        # if not all(scores[0] == score for score in scores):
        max_idx = np.argmax(scores)
        if max_idx == 0:
            maximum_trees.append(item)
        # if len(scores) == 4:
        #     if scores[0] > scores[1] and scores[0] > scores[2] \
        #     and scores[0] > scores[3]:
        #         maximum_trees.append(item)
        #     if scores[0] < scores[1] and scores[0] < scores[2] \
        #     and scores[0] < scores[3]:
        #         minimum_trees.append(item)

    f1s = [np.mean(f) for f in f1s]
    iou_bts = {k: np.mean(v) for k, v in zip(comb, iou_bts)}
    iou_tots = np.mean(iou_tots)

    n_biggers = {}
    n_equals = {}
    n_lowers = {}
    for k, v in pred_same_struct.items():
        bc = 0
        ec = 0
        lc = 0
        for e in v:
            if e["score1"] > e["score2"]:
                bc += 1
            elif e["score1"] < e["score2"]:
                lc += 1
            else:
                ec += 1
        n_biggers[k] = bc
        n_equals[k] = ec
        n_lowers[k] = lc

    # Accuracy of Fixed spans
    ratio_fixed = (correct_fixed + wrong_fixed) / sum(num_fluct)
    fixed_acc = correct_fixed / (correct_fixed + wrong_fixed)

    print(filtered_tree)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trees", nargs="+", required=True)
    parser.add_argument("-v", "--vocab", default=None)
    args = parser.parse_args()

    main(args)
