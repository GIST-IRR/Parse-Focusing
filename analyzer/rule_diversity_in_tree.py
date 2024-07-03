from collections import defaultdict, Counter
import argparse
import pickle
from copy import deepcopy
from itertools import repeat

from nltk import Tree
from utils import clean_rule


def main(
    filepath,
    vocab,
    output,
    binarization=False,
    horzMarkov=0,
    xbar=False,
    clean_label=False,
    clean_subtag=False,
    min_len=2,
    max_len=40,
):
    # Load trees
    trees = []
    with open(filepath, "r") as f:
        for l in f:
            trees.append(Tree.fromstring(l))
    # Tree binarization
    if binarization:
        for t in trees:
            t.chomsky_normal_form(
                factor="left",
                horzMarkov=horzMarkov,
                childChar="",
                parentChar="",
            )
            t.collapse_unary(collapsePOS=True)

    # Load vocab
    with open(vocab, "rb") as f:
        vocab = pickle.load(f)

    # Count unary productions
    length_count = defaultdict(int)
    rule_utilization = defaultdict(lambda: Counter())
    uniqueness_by_length = defaultdict(int)

    for t in trees:
        length = len(t.leaves())
        if length < min_len:
            continue
        elif length > max_len:
            continue

        length_count[length] += 1

        # Get productions
        prod = [
            p
            for p in t.productions()
            if not isinstance(p.rhs()[0], str) and p.lhs().symbol() != "ROOT"
            # if not isinstance(p.rhs()[0], str)
        ]

        n_prod = len(prod)
        if clean_label:
            prod = list(
                map(
                    clean_rule,
                    prod,
                    repeat(xbar, n_prod),
                    repeat(clean_subtag, n_prod),
                )
            )

        # Count rules
        tmp_prod = Counter(prod)
        prod = set(prod)
        rule_utilization[length].update(prod)
        uniqueness_by_length[length] += len(prod)

    # mean by length
    uniqueness_by_mean = {
        k: v / length_count[k] for k, v in sorted(uniqueness_by_length.items())
    }

    # rule에 따른 frequency로 재정렬
    total_ru = defaultdict(int)
    for l, d in rule_utilization.items():
        for r, v in d.items():
            total_ru[r] += v
    total_ru = {
        k: v
        for k, v in sorted(total_ru.items(), key=lambda x: x[1], reverse=True)
    }

    # parent에 따른 rule 분류
    rule_by_parent = defaultdict(set)
    for k in total_ru.keys():
        rule_by_parent[k.lhs()].add(k)
    len_rule_by_parent = {k: len(v) for k, v in rule_by_parent.items()}
    count_parents = len(len_rule_by_parent.keys())
    mean_utilized_rules = sum(len_rule_by_parent.values()) / count_parents

    # Counter to dict
    n_uniqueness_by_length = {}
    for length, counter in sorted(uniqueness_by_length.items()):
        total_count = counter.total()
        mass = 0
        n_dict = {}
        c = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for rule, count in c:
            count /= total_count
            mass += count
            if mass >= 0.95:
                break
            n_dict[rule] = count
        n_uniqueness_by_length[length] = n_dict

    for length, counter in uniqueness_by_length.items():
        uniqueness_by_length[length] = dict(counter)
        print(f"Length: {length}, Unique rules: {len(counter)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/data.clean/edit-english-train.txt",
    )
    parser.add_argument("--vocab", type=str, default="vocab/english.vocab")
    parser.add_argument("--output", type=str, default="gold_term_prob.pkl")
    parser.add_argument("--binarization", default=False, action="store_true")
    parser.add_argument("--horzMarkov", type=int, default=0)
    parser.add_argument("--xbar", default=False, action="store_true")
    parser.add_argument("--clean_label", default=False, action="store_true")
    parser.add_argument("--clean_subtag", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.filepath,
        args.vocab,
        args.output,
        args.binarization,
        args.horzMarkov,
        args.xbar,
        args.clean_label,
        args.clean_subtag,
    )
