# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric, UF1, LossMetric, UAS

from utils import (
    depth_from_tree,
    sort_span,
    span_to_tree,
    save_rule_heatmap,
    tensor_to_heatmap,
)


class CMD(object):
    def __call__(self, args):
        self.args = args

    def lambda_update(self, train_arg):
        if train_arg.get("dambda_warmup"):
            if self.iter < train_arg.warmup_start:
                dambda = 0
                # self.dambda = 1
            elif (
                self.iter >= train_arg.warmup_start
                and self.iter < train_arg.warmup_end
            ):
                dambda = 1 / (
                    1
                    + math.exp(
                        (-self.iter + train_arg.warmup_iter)
                        / (self.num_batch / 8)
                    )
                )
                # self.dambda = 1 - self.dambda
            else:
                # if not self.optim_flag:
                #     self.optimizer, self.optimizer_tmp = \
                #     self.optimizer_tmp, self.optimizer
                #     self.optim_flag = True
                dambda = 1
                # self.dambda = 0
        elif train_arg.get("dambda_step"):
            bound = train_arg.total_iter * train_arg.dambda_step
            if self.iter < bound:
                dambda = 0
                # self.dambda = 1
            else:
                # if not self.optim_flag:
                #     self.optimizer, self.optimizer_tmp = \
                #     self.optimizer_tmp, self.optimizer
                #     self.optim_flag = True
                dambda = 1
                # self.dambda = 0

            # self.dambda = self.model.entropy_rules(probs=True, reduce='mean')

            # if self.iter > self.num_batch * train_arg.warmup_start:
            #     ent = self.model.entropy_rules(probs=True, reduce='mean')
            #     factor = min(1, max(0, (self.iter-self.num_batch*train_arg.warmup_start)/(self.num_batch*train_arg.warmup_iter)))
            #     self.dambda = ent + factor * (1-ent)
            # else:
            #     self.dambda = 0

            # self.dambda = self.model.entropy_rules(probs=True).mean()

            # ent = self.model.entropy_rules(probs=True).mean()
            # factor = min(1, max(0, (self.iter-20000)/70000))
            # factor = min(1, self.iter/self.total_iter)
            # self.dambda = ent + factor * (1-ent)
        else:
            dambda = 1
            # self.dambda = 0
        return dambda

    def log_step(self, iter, start=0, step=500):
        """Log metrics for each logging step.
        Logging step check by **start** and **step**.

        Args:
            iter (int): current iteration.
            start (int, optional): start step of iteration. Defaults to 0.
            step (int, optional): each logging step. Defaults to 500.
        """
        if iter != start and iter % step == 0:
            self.writer.add_scalar(
                "train/loss", self.total_loss / step, iter
            )
            self.writer.add_scalar("train/lambda", self.dambda, iter)

            # metrics = self.model.metrics
            metrics = self.total_metrics
            for k, v in metrics.items():
                # self.writer.add_scalar(
                #     f"train/{k}", metrics[k].mean().item() / step, iter
                # )
                self.writer.add_scalar(
                    f"train/{k}", metrics[k] / step, iter
                )
                # tensor_to_heatmap(self.model.rules[k], dirname=heatmap_dir, filename=f'kl_term_{self.iter}.png')
                # self.writer.add_figure(f'train/{k}', tensor_to_heatmap(self.model.rules[k]), self.iter)

            self.writer.add_scalar(
                "train/rule_entropy",
                self.model.entropy("rule", probs=True, reduce="mean"),
                iter,
            )
            # initialize metrics
            self.total_loss = 0
            self.total_len = 0
            for k in metrics.keys():
                metrics[k] = 0
            
            if hasattr(self.model, "pf"):
                self.writer.add_histogram(
                    "train/partition_number",
                    self.model.pf.detach().cpu(),
                    iter,
                )
                self.pf = []

            for k, v in self.model.rules.items():
                self.writer.add_histogram(
                    f"train/{k}_prob", v.detach().cpu(), iter
                )
                if v.grad is not None:
                    self.writer.add_histogram(
                        f"train/{k}_grad", v.grad.detach().cpu(), iter
                    )

            ent = self.model.entropy("rule")
            self.writer.add_histogram(
                f"train/entropy", ent.detach().cpu(), iter
            )
            # save_rule_heatmap(self.model.rules, dirname='figure', filename=f'rule_{self.iter}.png', root=False, rule=False)
            # diff = {}
            # for k in self.model.rules.keys():
            #     diff[k] = self.model.rules[k] - prev_rules[k]
            # save_rule_heatmap(diff, dirname='figure', filename=f'rule_diff_{self.iter}.png', root=False, unary=True)
        return

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)
        train_arg = self.args.train
        heatmap_dir = os.path.join(self.args.save_dir, "heatmap")
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir, exist_ok=True)

        for x, _ in t:
            if not hasattr(train_arg, "warmup_epoch") and hasattr(
                train_arg, "warmup"
            ):
                if self.iter >= train_arg.warmup:
                    self.partition = True

            # Gradient zero
            self.optimizer.zero_grad()

            if self.partition:
                self.dambda = self.lambda_update(train_arg)

                # Soft gradients
                if self.dambda > 0:
                    loss, z_l = self.model.loss(
                        x, partition=self.partition, soft=True
                    )
                    t_loss = (loss + self.dambda * z_l).mean()
                    # self.model.soft_backward(
                    #     loss, z_l, self.optimizer,
                    #     dambda=self.dambda,
                    #     target="parameter"
                    # )
                else:
                    loss = self.model.loss(x)
                    z_l = None
                    t_loss = loss.mean()
                    
                t_loss.backward()

                loss = loss.mean()
                if z_l is not None:
                    z_l = z_l.mean()

            else:
                # Hard gradients
                loss = self.model.loss(x, partition=self.partition)
                t_loss = loss.mean()

                t_loss.backward()
                loss = loss.mean()

            # if 'prev_rules' not in locals():
            #     pass
            #     # save_rule_heatmap(self.model.rules, dirname='figure', filename=f'rule_gradient_{self.iter}.png', grad=True, root=False, unary=False)
            # else:
            #     diff = {}
            #     for k in self.model.rules.keys():
            #         diff[k] = self.model.rules[k] - prev_rules[k]
            #     save_rule_heatmap(diff, dirname='figure', filename=f'rule_diff_{self.iter}.png', root=False, unary=True)
            # prev_rules = self.model.rules
        
            # Gradient clipping
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), train_arg.clip
                )
            # Gradient update
            self.optimizer.step()

            # writer
            self.total_loss += loss.item()
            self.total_len += x["seq_len"].max().double()

            for k in self.total_metrics.keys():
                self.total_metrics[k] += self.model.metrics[k].mean().item()

            if hasattr(self.model, "pf"):
                self.pf = (
                    self.pf + self.model.pf.detach().cpu().tolist()
                    if self.model.pf.numel() != 1
                    else [self.model.pf.detach().cpu().tolist()]
                )

            if getattr(train_arg, "heatmap"):
                if self.iter % int(self.total_iter / 10) == 0:
                    self.model.save_rule_heatmap(
                        dirname=heatmap_dir,
                        filename=f"rule_dist_{self.iter}.png",
                    )
            
            self.log_step(self.iter, start=0, step=500)

            # prev_rules = self.model.rules
            # Check total iteration
            self.iter += 1
        return

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        eval_dep=False,
        decode_type="mbr",
        model=None,
        eval_depth=False,
        left_binarization=False,
        right_binarization=False,
    ):
        if model == None:
            model = self.model
        model.eval()

        metric_f1 = UF1()
        metric_f1_left = UF1()
        metric_f1_right = UF1()
        metric_uas = UAS()
        metric_ll = LikelihoodMetric()

        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)
        print("decoding mode:{}".format(decode_type))
        print("evaluate_dep:{}".format(eval_dep))

        depth = (
            self.args.test.depth - 2 if hasattr(self.args.test, "depth") else 0
        )

        self.pf_sum = torch.zeros(depth + 1)
        self.estimated_depth = {}
        self.estimated_depth_by_length = {}
        self.parse_trees = []
        for x, y in t:
            result = model.evaluate(
                x, decode_type=decode_type, eval_dep=eval_dep, depth=depth
            )

            result["prediction"] = sort_span(result["prediction"])
            self.parse_trees += [
                {
                    "word": x["word"][i].tolist(),
                    "gold_tree": y["gold_tree"][i],
                    "pred_tree": result["prediction"][i],
                }
                for i in range(x["word"].shape[0])
            ]
            # for tree in predicted_trees:
            #     for i, pos in enumerate(tree.treepositions('leaves')):
            #         tree[pos] = y['pos'][i]
            predicted_trees = [span_to_tree(r) for r in result["prediction"]]
            s_depth = [depth_from_tree(t) for t in predicted_trees]
            for d in s_depth:
                if d in self.estimated_depth:
                    self.estimated_depth[d] += 1
                else:
                    self.estimated_depth[d] = 1
            for i, l in enumerate(x["seq_len"]):
                l = l.item()
                d = s_depth[i]
                if l in self.estimated_depth_by_length:
                    if d in self.estimated_depth_by_length[l]:
                        self.estimated_depth_by_length[l][d] += 1
                    else:
                        self.estimated_depth_by_length[l][d] = 1
                else:
                    self.estimated_depth_by_length[l] = {}
                    self.estimated_depth_by_length[l][d] = 1

            if eval_depth:
                if len(result["prediction"][0][0]) >= 3:
                    metric_f1(
                        result["prediction"],
                        y["gold_tree"],
                        y["depth"],
                        lens=True,
                        nonterminal=True,
                    )
                else:
                    metric_f1(
                        result["prediction"],
                        y["gold_tree"],
                        y["depth"],
                        lens=True,
                    )
            else:
                if len(result["prediction"][0][0]) >= 3:
                    metric_f1(
                        result["prediction"],
                        y["gold_tree"],
                        lens=True,
                        nonterminal=True,
                    )
                else:
                    metric_f1(result["prediction"], y["gold_tree"], lens=True)
            if left_binarization:
                if len(result["prediction"][0][0]) >= 3:
                    metric_f1_left(
                        result["prediction"],
                        y["gold_tree_left"],
                        y["depth_left"],
                        lens=True,
                        nonterminal=True,
                    )
                else:
                    metric_f1_left(
                        result["prediction"],
                        y["gold_tree_left"],
                        y["depth_left"],
                        lens=True,
                    )
            if right_binarization:
                if len(result["prediction"][0][0]) >= 3:
                    metric_f1_right(
                        result["prediction"],
                        y["gold_tree_right"],
                        y["depth_right"],
                        lens=True,
                        nonterminal=True,
                    )
                else:
                    metric_f1_right(
                        result["prediction"],
                        y["gold_tree_right"],
                        y["depth_right"],
                        lens=True,
                    )

            self.pf_sum = (
                self.pf_sum + torch.sum(result["depth"], dim=0).detach().cpu()
            )
            metric_ll(result["partition"], x["seq_len"])
            if eval_dep:
                metric_uas(result["prediction_arc"], y["head"])

        return (
            metric_f1,
            metric_uas,
            metric_ll,
            metric_f1_left,
            metric_f1_right,
        )
        # if not eval_dep:
        #     return metric_f1, metric_ll
        # else:
        #     return metric_f1, metric_uas, metric_ll
