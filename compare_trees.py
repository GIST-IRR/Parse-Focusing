import pickle
from pathlib import Path
import argparse
from collections import Counter
from utils import span_to_list, span_to_tree

def open_pickle(filename):
    with Path(filename).open('rb') as f:
        obj = pickle.load(f)
    return obj

def f1(pred, gold):
    eps = 1e-8
    # in the case of sentence length=1
    if len(pred) == 0:
        return None
    length = max(gold,key=lambda x:x[1])[1]
    #removing the trival span
    gold = list(filter(lambda x: x[0]+1 != x[1], gold))
    pred = list(filter(lambda x: x[0]+1 != x[1], pred))
    #remove the entire sentence span.
    gold = list(filter(lambda x: not (x[0]==0 and x[1]==length), gold))
    pred = list(filter(lambda x: not (x[0]==0 and x[1]==length), pred))
    #remove label.
    gold = [g[:2] for g in gold]
    pred = [p[:2] for p in pred]
    gold = list(map(tuple, gold))

    gold = set(gold)
    pred = set(pred)
    overlap = pred.intersection(gold)
    prec = float(len(overlap)) / (len(pred) + eps)
    reca = float(len(overlap)) / (len(gold) + eps)
    if len(gold) == 0:
        reca = 1.
        if len(pred) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

def set_leaves(tree, words):
    tp = tree.treepositions('leaves')
    for i, p in enumerate(tp):
        tree[p] = str(words[i])

def pretty_print(tree):
    word = tree['word']
    gold = span_to_tree(tree['gold'])
    pred1 = span_to_tree(tree['pred1'])
    pred2 = span_to_tree(tree['pred2'])

    set_leaves(gold, word)
    set_leaves(pred1, word)
    set_leaves(pred2, word)

    gold.pretty_print()
    pred1.pretty_print()
    pred2.pretty_print()

def main(args):
    tree1 = open_pickle(args.trees1)
    tree2 = open_pickle(args.trees2)

    vocab = tree1['vocab']
    tree1 = tree1['trees']
    tree2 = tree2['trees']

    tree1 = sorted(tree1, key=lambda x: x['word'])
    tree2 = sorted(tree2, key=lambda x: x['word'])

    def decode(tree):
        word_idx = tree['word']
        word = []
        for w in word_idx:
            word.append(vocab.to_word(w))
        tree['word'] = word

    for t in tree1:
        decode(t)
    for t in tree2:
        decode(t)

    smaller = []
    same = []
    larger = []

    for t1, t2 in zip(tree1, tree2):
        word = t1['word']
        gold = t1['gold_tree']
        pred1 = t1['pred_tree']
        pred2 = t2['pred_tree']

        t1_score = f1(pred1, gold)
        t2_score = f1(pred2, gold)

        item = {
            'word': word,
            'gold': gold,
            'pred1': pred1,
            'pred2': pred2,
            'p1_f1': t1_score,
            'p2_f2': t2_score
        }
        if t1_score < t2_score:
            smaller.append(item)
        elif t1_score > t2_score:
            larger.append(item)
        else:
            same.append(item)

    # s_unk = Counter([s['word'].count(1) for s in smaller])
    # l_unk = Counter([l['word'].count(1) for l in larger])
    smaller = sorted(smaller, key=lambda x: len(x['word']))
    larger = sorted(larger, key=lambda x: len(x['word']))
    # tree = span_to_tree(larger[4]['gold'])
    pretty_print(larger[15])
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trees1')
    parser.add_argument('--trees2')
    args = parser.parse_args()

    main(args)