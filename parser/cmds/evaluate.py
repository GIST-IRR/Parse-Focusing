# -*- coding: utf-8 -*-


from parser.cmds.cmd import CMD

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
import click
from utils import (
    span_to_tree,
    save_rule_heatmap,
    save_rule_heatmap_raw,
    tensor_to_heatmap,
    save_correspondence,
    visualize_embeddings,
    visualize_probability_distribution,
    visualize_graph,
    visualize_rule_graph,
    count_recursive_rules,
    save_rule_distribution_raw,
    outlier,
)

import random
from pathlib import Path
import pickle
import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
import torch_support.reproducibility as reproducibility
from torch_support.load_model import (
    get_model_args,
    get_optimizer_args,
    set_model_dir,
)


class Evaluate(CMD):
    def __call__(
        self,
        args,
        eval_dep=False,
        decode_type="mbr",
        data_split="test",
        tag="best",
    ):
        super(Evaluate, self).__call__(args)
        self.device = args.device
        self.args = args

        dataset = DataModule(args)
        self.vocab = dataset.word_vocab

        # dataset = DataModule(args)
        args.model.update({"V": len(dataset.word_vocab)})
        set_model_dir("parser.model")
        self.model = get_model_args(args.model, self.device)

        best_model_path = self.args.load_from_dir + f"/{tag}.pt"
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print("successfully load")

        eval_depth = getattr(self.args.test, "eval_depth", False)
        left_binarization = getattr(self.args.test, "left_binarization", False)
        right_binarization = getattr(
            self.args.test, "right_binarization", False
        )

        if data_split == "train":
            test_loader = dataset.train_dataloader()
        elif data_split == "valid":
            test_loader = dataset.val_dataloader()
        elif data_split == "test":
            test_loader = dataset.test_dataloader

        test_loader_autodevice = DataPrefetcher(
            test_loader, device=self.device
        )

        (
            metric_f1,
            metric_uas,
            likelihood,
            metric_left,
            metric_right,
        ) = self.evaluate(
            test_loader_autodevice,
            eval_dep=eval_dep,
            decode_type=decode_type,
            eval_depth=eval_depth,
            left_binarization=left_binarization,
            right_binarization=right_binarization,
        )
        try:
            print(metric_uas)
        except:
            print("No UAS")
        print(metric_f1)
        print(likelihood)

        for k in ["SBAR", "NP", "VP", "PP", "ADJP", "ADVP"]:
            print(f"{k}: {metric_f1.label_recall.get(k, 0)}")

        # Log - Tensorboard
        self.writer = SummaryWriter(self.args.load_from_dir)
        for i, pf in enumerate(self.pf_sum):
            self.writer.add_scalar(
                f"test/{data_split}/partition_number", pf / metric_f1.n, i
            )
        for k, v in metric_f1.sentence_uf1_d.items():
            self.writer.add_scalar(f"test/{data_split}/f1_depth", v, k)
        # for k, v in metric_f1.sentence_uf1_l.items():
        #     self.writer.add_scalar('test/f1_length', v, k)
        for k, v in metric_f1.sentence_ex_d.items():
            self.writer.add_scalar(f"test/{data_split}/Ex_depth", v, k)

        # F1 score for each depth
        for k, v in metric_left.sentence_uf1_d.items():
            self.writer.add_scalar(f"test/{data_split}/f1_left_depth", v, k)
        # F1 score for each length
        # for k, v in metric_left.sentence_uf1_l.items():
        #     self.writer.add_scalar('valid/f1_left_length', v, k)
        for k, v in metric_left.sentence_ex_d.items():
            self.writer.add_scalar(f"test/{data_split}/Ex_left_depth", v, k)

        # F1 score for each depth
        for k, v in metric_right.sentence_uf1_d.items():
            self.writer.add_scalar(f"test/{data_split}/f1_right_depth", v, k)
        # F1 score for each length
        # for k, v in metric_right.sentence_uf1_l.items():
        #     self.writer.add_scalar('valid/f1_right_length', v, k)
        for k, v in metric_right.sentence_ex_d.items():
            self.writer.add_scalar(f"test/{data_split}/Ex_right_depth", v, k)

        self.estimated_depth = dict(sorted(self.estimated_depth.items()))
        for k, v in self.estimated_depth.items():
            self.writer.add_scalar(
                f"test/{data_split}/estimated_depth", v / metric_f1.n, k
            )

        self.estimated_depth_by_length = dict(
            sorted(self.estimated_depth_by_length.items())
        )
        for k in self.estimated_depth_by_length.keys():
            self.estimated_depth_by_length[k] = dict(
                sorted(self.estimated_depth_by_length[k].items())
            )
        for k, v in self.estimated_depth_by_length.items():
            total = sum(v.values())
            for d, n in v.items():
                self.writer.add_scalar(
                    f"test/{data_split}/predicted_depth_by_length_{k}",
                    n / total,
                    d,
                )

        self.writer.flush()
        self.writer.close()

        if hasattr(self, "parse_trees"):
            pred_trees = [
                span_to_tree(t["pred_tree"]) for t in self.parse_trees
            ]
            gold_trees = [
                span_to_tree(t["gold_tree"]) for t in self.parse_trees
            ]
            gd_rule_ct, gd_bi_ct, gd_rec_ct = count_recursive_rules(gold_trees)
            pd_rule_ct, pd_bi_ct, pd_rec_ct = count_recursive_rules(pred_trees)
            print("Ratio of gold recursive rule:")
            print(f"\tnumber of nonterminal rules: {gd_bi_ct.total()}")
            print(f"\tnumber of recursive rules: {gd_rec_ct.total()}")
            print(
                f"\tratio of recursive rules: {gd_rec_ct.total() / gd_bi_ct.total()}"
            )
            print("Ratio of predicted recursive rule:")
            print(f"\tnumber of nonterminal rules: {pd_bi_ct.total()}")
            print(f"\tnumber of recursive rules: {pd_rec_ct.total()}")
            print(
                f"\tratio of recursive rules: {pd_rec_ct.total() / pd_bi_ct.total()}"
            )

        if hasattr(metric_f1, "correspondence"):
            save_correspondence(
                metric_f1.correspondence, dirname=self.args.load_from_dir
            )

        try:
            visualize_embeddings(
                self.model,
                dirname=self.args.load_from_dir,
                # label=False
            )
        except:
            pass

        try:
            visualize_probability_distribution(
                self.model.nonterms().exp(),
                dirname=self.args.load_from_dir,
            )
        except:
            pass

        try:
            pred_trees = [
                span_to_tree(t["pred_tree"]) for t in self.parse_trees
            ]
            # pred_trees = [t for t in pred_trees if len(t.leaves()) == 3]
            visualize_graph(pred_trees, dirname=self.args.load_from_dir)
            visualize_rule_graph(pred_trees, dirname=self.args.load_from_dir)
        except:
            pass

        # Heatmap
        try:
            save_rule_heatmap_raw(
                self.model,
                dirname=self.args.load_from_dir,
                filename="eval_rule_dist.png",
                root=False,
                rule=False,
            )
        except:
            pass

        # Distribution
        try:
            save_rule_distribution_raw(
                self.model,
                dirname=self.args.load_from_dir,
                filename="eval_rule_distribution.png",
                root=False,
            )
            outliers = outlier(self.model.rules["rule"][0].cpu().numpy())
        except:
            pass

        # Log - CSV
        import csv

        label_recall = getattr(self.args.test, "label_recall", False)
        if label_recall:
            nt_log = os.path.join(self.args.load_from_dir, "label_recall.csv")
            with open(nt_log, "w", newline="") as f:
                writer = csv.writer(f)
                for label, recall in metric_f1.label_recall.items():
                    writer.writerow([label, recall])

        # log_file = os.path.join(os.path.dirname(args.load_from_dir), 'test.csv')
        log_dir = Path(args.load_from_dir).parent
        log_file = log_dir / (str(log_dir.name) + ".csv")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    args.load_from_dir,
                    metric_f1.sentence_uf1,
                    metric_f1.corpus_uf1,
                    likelihood.avg_likelihood.item(),
                    likelihood.perplexity.item(),
                ]
            )

        # heatmap_dir = Path(self.args.save_dir) / 'heatmap'
        # tensor_to_heatmap(
        #     self.model.rules,
        #     dirname=heatmap_dir,
        #     filename=f'rule_dist_{data_split}.png'
        # )

        tree_dir = Path(args.load_from_dir)
        tree_file = tree_dir / f"{data_split}-parse_trees.pkl"
        tree_obj = {"vocab": self.vocab, "trees": self.parse_trees}
        with tree_file.open("wb") as f:
            pickle.dump(tree_obj, f)
