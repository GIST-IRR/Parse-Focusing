import pickle
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import itertools

import numpy as np
import torch
from nltk import Tree

from utils import load_trees, span_to_tree, tree_to_span, sort_span, clean_word
from torch_support.metric import preprocess_span
from torch_support.metric import sentence_level_f1 as f1
from torch_support.metric import iou as IoU


def set_leaves(tree, words):
    tp = tree.treepositions("leaves")
    for i, p in enumerate(tp):
        tree[p] = str(words[i])


def pretty_print(tree):
    word = tree["word"]

    def _pretty_print(tree, word):
        t = span_to_tree(tree)
        set_leaves(t, word)
        t.pretty_print()

    _pretty_print(tree["gold"], word)
    _pretty_print(tree["pred1"], word)
    _pretty_print(tree["pred2"], word)


def main(args, max_len=40, min_len=2):
    # Load vocab
    if args.vocab is not None:
        with args.vocab.open("rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = None
        raise ValueError("Vocab is required.")

    # Load gold parse trees
    if args.gold is not None:
        args.eval_gold = True
        gold = load_trees(args.gold, min_len, max_len, True, vocab)

    # Load predicted parse trees
    trees = [load_trees(t, min_len, max_len, True, vocab) for t in args.trees]

    comb = list(itertools.combinations(range(len(trees)), 2))
    # Compare trees
    filtered_tree = []
    totally_same = []
    different = []
    diff_n_bigger = []
    f1s = [[] for _ in range(len(trees))]
    iou_bts = [[] for _ in range(len(comb))]
    self_f1s = [[] for _ in range(len(comb))]
    iou_tots = []

    pred_same_struct = defaultdict(list)

    maximum_trees = []
    minimum_trees = []

    # Count common & uncommon rules
    cuc_bins = []

    # Count unknown tokens
    n_unk = [0 for _ in range(len(trees))]

    for i, ts in enumerate(zip(*trees)):
        word = ts[0]["word"]
        sentence = ts[0]["sentence"]

        # Check all trees are same
        assert all(t["word"] == word for t in ts)

        # Preprocess gold trees
        if args.eval_gold:
            gs = gold[i]
            # Check predicted trees and gold tree are same
            assert gs["sentence"] == sentence

            gt = gs["span"]
            gt = [s[:2] for s in gt]
            gt = sort_span(gt)

        # Preprocessing predicted trees
        pred = [[s[:2] for s in t["span"]] for t in ts]
        # Add root and trivial spans
        for p in pred:
            if len(p) < len(word):
                p.append((0, len(word)))
                for i in range(len(word)):
                    p.append((i, i + 1))
        pred = list(map(sort_span, pred))

        # Count number of unknown tokens
        n_u = [sum([1 for w in t["word"] if w == "<unk>"]) for t in ts]
        for j, n in enumerate(n_u):
            n_unk[j] += n

        # Evaluate with gold
        if args.eval_gold:
            # Evaluation F1 for gold
            scores = [f1(p, gt) for p in pred]
            # Evaluation IoU for gold
            iou = [IoU(p, gt) for p in pred]
            # Append to result list
            for f, s in zip(f1s, scores):
                f.append(s)

        # Evaluation IoU between predictions
        # # Compare each pair of trees
        iou_bt = [IoU(pred[e[0]], pred[e[1]]) for e in comb]
        for ib, s in zip(iou_bts, iou_bt):
            ib.append(s)
        # # Compare whole trees once
        # iou_tot = IoU(*pred)
        # iou_tots.append(iou_tot)

        # Evaluation self F1 between predictions
        self_f1 = [f1(pred[e[0]], pred[e[1]]) for e in comb]
        for sf, s in zip(self_f1s, self_f1):
            sf.append(s)

        # Evaluation shared, common, (whatever...) pare evalution
        if args.eval_gold:
            # common and uncommon spans in predicted trees
            p_gold = preprocess_span(gt, len(word))
            union = set.union(*[set(p) for p in pred])
            union = list(preprocess_span(union, len(word)))
            union_count = {}
            for e in union:
                exist_count = 0
                for p in pred:
                    if e in p:
                        exist_count += 1
                correct = 1 if e in p_gold else 0
                union_count[e] = [correct, exist_count]
            cuc_bins.append(union_count)

        item = {
            "word": word,
            "gold": gt,
            "mean": np.mean(scores),
            "var": np.var(scores),
        }

        # Add items for predicted trees
        for i, p in enumerate(pred):
            # item[f"word{i+1}"] = ts[i]["word"]
            item[f"pred{i+1}"] = p
            item[f"score{i+1}"] = scores[i]
            item[f"iou{i+1}"] = iou[i]
            # if len(pred) <= 2 and i == 1:
            #     continue
            # else:
            #     item[f"iou_bt{i+1}"] = iou_bt[i]

        # If variance is bigger than 0, add to filtered tree
        # It choose the item that has different predicted trees.
        if item["var"] > 0.0:
            filtered_tree.append(item)

        # If all predicted trees are same, add to totally_same
        # If not, add to different
        if all(pred[0] == p for p in pred):
            totally_same.append(item)
        else:
            different.append(item)

        # if the score of first tree is bigger than the second tree
        # and the length of word is bigger than 2, add to diff_n_bigger
        if len(word) > 2 and scores[0] > scores[1]:
            diff_n_bigger.append(item)

        # Save the structure of the same predicted treess
        if len(word) > 2:
            k = frozenset([tuple(e[:2]) for e in gt])
            pred_same_struct[k].append(item)

        # Maximum trees
        # if not all(scores[0] == score for score in scores):
        max_idx = np.argmax(scores)
        if max_idx == 0:
            maximum_trees.append(item)

    f1s = [np.mean(f) for f in f1s]
    iou_bts = {k: np.mean(v) for k, v in zip(comb, iou_bts)}
    iou_tots = np.mean(iou_tots)

    filtered_diff_n_bigger = [
        e for e in diff_n_bigger if not "<unk>" in e["word"]
    ]

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

    # Accuracy of Fixed spans: common uncommon에 대한 내용으로 수정 및 정리
    # ratio_fixed = (correct_fixed + wrong_fixed) / sum(num_fluct)
    # fixed_acc = correct_fixed / (correct_fixed + wrong_fixed)

    print(filtered_tree)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trees", nargs="+", required=True, type=Path)
    parser.add_argument("-g", "--gold", default=None, type=Path)
    parser.add_argument("-v", "--vocab", default=None, type=Path)
    args = parser.parse_args()

    main(args)
