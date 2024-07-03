from collections import defaultdict
import argparse
from pathlib import Path
import pickle

import numpy as np
import torch

from nltk import Tree
from utils import clean_word, clean_symbol


def main(filepath, vocab, output, horzMarkov=0, normalize=False, xbar=False):
    # Load trees
    trees = []
    with open(filepath, "r") as f:
        for l in f:
            trees.append(Tree.fromstring(l))
    # Tree binarization
    for t in trees:
        t.chomsky_normal_form(
            horzMarkov=horzMarkov, childChar="", parentChar=""
        )
        t.collapse_unary(collapsePOS=True)

    # Load vocab
    with open(vocab, "rb") as f:
        vocab = pickle.load(f)

    # Count unary productions
    unary_prod = defaultdict(lambda: defaultdict(int))
    binary_prod = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    root_prod = defaultdict(int)
    preterminal_symbols = set()
    nonterminal_symbols = set()
    binary_freq = defaultdict(int)

    unk_pos = defaultdict(int)

    for t in trees:
        if len(t.leaves()) == 1:
            continue
        elif len(t.leaves()) > 40:
            continue
        unary_pos = t.treepositions("leaves")
        unaries = [t[p[:-1]].productions()[0] for p in unary_pos]

        for u in unaries:
            pt = clean_symbol(u.lhs().symbol(), xbar=xbar)
            preterminal_symbols.add(pt)

            w = clean_word([u.rhs()[0]])[0]
            if w not in vocab:
                unk_pos[pt] += 1
                w = "<unk>"
            w = vocab[w]
            unary_prod[pt][w] += 1

        binaries = [p for p in t.productions() if len(p.rhs()) == 2]

        for b in binaries:
            pt = clean_symbol(b.lhs().symbol(), xbar=xbar)
            l = clean_symbol(b.rhs()[0].symbol(), xbar=xbar)
            r = clean_symbol(b.rhs()[1].symbol(), xbar=xbar)
            nonterminal_symbols.add(pt)
            nonterminal_symbols.add(l)
            nonterminal_symbols.add(r)

            binary_prod[pt][l][r] += 1
            binary_freq[(pt, l, r)] += 1

        root = t.productions()[0]
        root_rhs_symbol = clean_symbol(root.rhs()[0].symbol(), xbar=xbar)
        nonterminal_symbols.add(root_rhs_symbol)
        root_prod[root_rhs_symbol] += 1

    nonterminal_symbols = nonterminal_symbols - preterminal_symbols
    n_nonterm = len(nonterminal_symbols)
    n_preterm = len(preterminal_symbols)

    n_total = n_nonterm + n_preterm
    l_nonterm = sorted(list(nonterminal_symbols))
    l_preterm = sorted(list(preterminal_symbols))
    total_symbols = l_nonterm + l_preterm
    total_symbols = {s: i for i, s in enumerate(total_symbols)}

    binary_freq = sorted(binary_freq.items(), key=lambda x: x[1], reverse=True)

    # Normalize
    unary_prob = np.zeros([n_preterm, len(vocab)])
    for p, c in unary_prod.items():
        symbol_idx = total_symbols[p] - n_nonterm
        if normalize:
            total = sum(c.values())
            for w in c.keys():
                # c[w] /= total
                unary_prob[symbol_idx, w] = c[w] / total
        else:
            for w in c.keys():
                unary_prob[symbol_idx, w] = c[w]

    # Normalize
    binary_prob = np.zeros([n_nonterm, n_total, n_total])
    for p, lr in binary_prod.items():
        p_idx = total_symbols[p]
        if normalize:
            total = sum([sum(e.values()) for e in lr.values()])
            for l, rc in lr.items():
                l_idx = total_symbols[l]
                for r, c in rc.items():
                    r_idx = total_symbols[r]
                    binary_prob[p_idx, l_idx, r_idx] = c / total
        else:
            for l, rc in lr.items():
                l_idx = total_symbols[l]
                for r, c in rc.items():
                    r_idx = total_symbols[r]
                    binary_prob[p_idx, l_idx, r_idx] = c

    # Normalize
    root_prob = np.zeros([1, n_nonterm])
    if normalize:
        total = sum(root_prod.values())
        for p, c in root_prod.items():
            p_idx = total_symbols[p]
            root_prob[0, p_idx] = c / total
    else:
        for p, c in root_prod.items():
            p_idx = total_symbols[p]
            root_prob[0, p_idx] = c

    # Save
    output = Path(output)
    output_stem = output.stem
    flag = "prob" if normalize else "freq"

    def save_prob(prob, path):
        with path.open("wb") as f:
            pickle.dump(prob, f)
        torch.save(prob, path.with_suffix(".pt"))

    unary_path = output.with_stem(output_stem + f"_unary_{flag}")
    save_prob(unary_prob, unary_path)
    binary_path = output.with_stem(output_stem + f"_binary_{flag}")
    save_prob(binary_prob, binary_path)
    root_path = output.with_stem(output_stem + f"_root_{flag}")
    save_prob(root_prob, root_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/data.clean/edit-english-train.txt",
    )
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument("--output", type=str, default="gold_term_prob.pkl")
    parser.add_argument("--horzMarkov", type=int, default=0)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--xbar", type=bool, default=False)
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.horzMarkov,
        args.normalize,
        args.xbar,
    )
