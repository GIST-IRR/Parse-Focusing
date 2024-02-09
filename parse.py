# -*- coding: utf-8 -*-

import os
from parser.cmds import Evaluate
import torch
from easydict import EasyDict as edict

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import click
import random
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from nltk import Tree

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.batch import DataSetIter
from torch.utils.data import Sampler

from parser.helper.loader_wrapper import DataPrefetcher
from parser.helper.metric import LikelihoodMetric, UF1

from torch_support.load_model import (
    get_model_args,
    get_optimizer_args,
    set_model_dir,
)

from utils import (
    tree_to_span,
    depth_from_tree,
    sort_span,
    span_to_tree,
)


class ByLengthSampler(Sampler):
    def __init__(self, dataset, batch_size=4):
        self.group = defaultdict(list)
        self.seq_lens = dataset["seq_len"]
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size = batch_size
        total = []

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for idx, lst in self.group.items():
            total = total + list(chunks(lst, self.batch_size))
        self.total = total

    def __iter__(self):
        random.shuffle(self.total)
        for batch in self.total:
            yield batch

    def __len__(self):
        return len(self.total)


def get_dataset(filepath, vocab):
    def clean_word(words):
        import re

        def clean_number(w):
            new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
            return new_w

        return [clean_number(word.lower()) for word in words]

    dataset = DataSet()
    data = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            tree = Tree.fromstring(line)
            data["word"].append(tree.leaves())
            data["gold_tree"].append(tree_to_span(tree))
            data["tree_str"].append(line.strip())

    dataset.add_field("sentence", data["word"], ignore_type=True)
    dataset.add_field(
        "gold_tree", data["gold_tree"], padder=None, ignore_type=True
    )
    dataset.add_field(
        "tree_str", data["tree_str"], padder=None, ignore_type=True
    )
    dataset.add_seq_len(field_name="sentence", new_field_name="seq_len")
    dataset.apply_field(clean_word, "sentence", "word")

    vocab.index_dataset(dataset, field_name="word")

    dataset = dataset.drop(lambda x: x["seq_len"] == 1, inplace=True)
    dataset.set_input("word", "seq_len")
    dataset.set_target("sentence", "gold_tree", "tree_str")
    return dataset


@click.command()
@click.option("--filepath", required=True)
@click.option("--output", required=True)
@click.option("--vocab", required=True)
@click.option("--batch_size", default=4)
@click.option(
    "--eval_dep",
    default=False,
    help="evaluate dependency, only for N(B)L-PCFG",
)
@click.option("--data_split", default="test")
@click.option("--decode_type", default="mbr", help="viterbi or mbr")
@click.option("--load_from_dir", default="")
@click.option("--device", "-d", default="0")
@click.option("--tag", "-t", default="best")
def main(
    filepath,
    output,
    vocab,
    load_from_dir,
    batch_size,
    eval_dep,
    data_split,
    decode_type,
    device,
    tag,
):
    yaml_cfg = load(open(load_from_dir + "/config.yaml", "r"), Loader=Loader)
    args = edict(yaml_cfg)

    print(f"Set the device with ID {device} visible")
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    word_vocab = pickle.load(open(vocab, "rb"))
    dataset = get_dataset(filepath, word_vocab)

    args.model.update({"V": len(word_vocab)})
    set_model_dir("parser.model")
    model = get_model_args(args.model, device)

    best_model_path = load_from_dir + f"/{tag}.pt"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print("successfully load")

    sampler = ByLengthSampler(dataset=dataset, batch_size=batch_size)
    dataloader = DataSetIter(dataset=dataset, batch_sampler=sampler)
    autoloader = DataPrefetcher(dataloader, device=device)

    with torch.no_grad():
        model.eval()

        metric_f1 = UF1()
        metric_ll = LikelihoodMetric()

        t = tqdm(
            autoloader, total=int(len(autoloader)), position=0, leave=True
        )
        print("decoding mode:{}".format(decode_type))
        print("evaluate_dep:{}".format(eval_dep))

        parse_trees = []
        for x, y in t:
            result = model.evaluate(
                x,
                decode_type=decode_type,
                eval_dep=eval_dep,
            )
            # Save predicted parse trees
            result["prediction"] = sort_span(result["prediction"])
            parse_trees += [
                {
                    "sentence": y["sentence"][i],
                    "word": x["word"][i].tolist(),
                    "gold_tree": y["gold_tree"][i],
                    "pred_tree": result["prediction"][i],
                }
                for i in range(x["word"].shape[0])
            ]
            nonterminal = len(result["prediction"][0][0]) >= 3
            metric_f1(
                result["prediction"],
                y["gold_tree"],
                lens=True,
                nonterminal=nonterminal,
            )

            metric_ll(result["partition"], x["seq_len"])

    print(metric_f1)
    print(metric_ll)

    output_tree = []
    for data in dataset["sentence"]:
        pred = list(filter(lambda x: x["sentence"] == data, parse_trees))[0]
        pred_tree = span_to_tree(pred["pred_tree"])
        word_pos = pred_tree.treepositions("leaves")
        for i, p in enumerate(word_pos):
            pred_tree[p] = pred["sentence"][i]
        output_tree.append(pred_tree._pformat_flat("", "()", False))

    # for t in parse_trees:
    #     idx = sentences.index(t["sentence"])
    #     pred_tree = span_to_tree(t["pred_tree"])
    #     word_pos = pred_tree.treepositions("leaves")
    #     for i, p in enumerate(word_pos):
    #         pred_tree[p] = t["sentence"][i]
    #     output_tree[idx] = pred_tree._pformat_flat("", "()", False)

    output_path = Path(output)
    output_path.write_text("\n".join(output_tree))
    gold_path = output_path.with_suffix(".gold.txt")
    gold_path.write_text("\n".join(dataset["tree_str"]))
    exit()


if __name__ == "__main__":
    main()
