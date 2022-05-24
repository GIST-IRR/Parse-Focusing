# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

from utils import depth_from_span

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train

        for x, _ in t:
            if not hasattr(train_arg, 'warmup_epoch') and hasattr(train_arg, 'warmup'):
                if self.iter >= train_arg.warmup:
                    self.partition = True

            self.optimizer.zero_grad()
            
            if self.partition \
                and hasattr(train_arg, 'soft_loss_target') \
                and hasattr(train_arg, 'soft_loss_mode'):
                # Split loss
                loss, z_l = self.model.loss(x, partition=self.partition, soft=True)
                self.model.soft_backward(
                    loss, z_l, self.optimizer,
                    target=train_arg.soft_loss_target,
                    mode=train_arg.soft_loss_mode
                )
            else:
                # Original
                loss = self.model.loss(x, partition=self.partition)
                loss.backward()
            
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            # writer
            self.total_loss += loss.item()

            if hasattr(self.model, 'pf'):
                self.pf = self.pf + self.model.pf.detach().cpu().tolist() if self.model.pf.numel() != 1 else [self.model.pf.detach().cpu().tolist()]
            if self.iter != 0 and self.iter % 1000 == 0:
                self.writer.add_scalar('train/loss', self.total_loss/1000, self.iter)
                self.total_loss = 0
                if hasattr(self.model, 'pf'):
                    self.writer.add_histogram('train/partition_number', self.model.pf.detach().cpu(), self.iter)
                    self.pf = []
                # if hasattr(self.model.rules, 'root'):
                #     self.writer.add_histogram('train/root_prob', self.model.rules['root'].reshape(x['word'].shape[0], -1).detach().cpu(), self.iter)
                # if hasattr(self.model.rules, 'rule'):
                #     self.writer.add_histogram('train/rule_prob', self.model.rules['rule'].reshape(x['word'].shape[0], -1).detach().cpu(), self.iter)
            # Check total iteration
            self.iter += 1

        return


    @torch.no_grad()
    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None,
                    eval_depth=False, left_binarization=False, right_binarization=False):
        if model == None:
            model = self.model
        model.eval()

        metric_f1 = UF1()
        metric_f1_o = UF1()
        metric_f1_left = UF1()
        metric_f1_right = UF1()
        metric_uas = UAS()
        metric_ll = LikelihoodMetric()

        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))

        depth = self.args.test.depth - 2 if hasattr(self.args.test, 'depth') else 0

        self.pf_sum = torch.zeros(depth + 1)
        self.estimated_depth = {}
        self.estimated_depth_by_length = {}
        for x, y in t:
            result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep, depth=depth)

            s_depth = [depth_from_span(r) for r in result['prediction']]
            for d in s_depth:
                if d in self.estimated_depth:
                    self.estimated_depth[d] += 1
                else:
                    self.estimated_depth[d] = 1
            for i, l in enumerate(x['seq_len']):
                l = l.item()
                d = s_depth[i]
                if l in self.estimated_depth_by_length:
                    if d in self.estimated_depth_by_length[l]:
                        self.estimated_depth_by_length[l][d] +=1
                    else:
                        self.estimated_depth_by_length[l][d] = 1
                else:
                    self.estimated_depth_by_length[l] = {}
                    self.estimated_depth_by_length[l][d] = 1

            if eval_depth:
                metric_f1(result['prediction'], y['gold_tree'], y['depth'], lens=True, nonterminal=True)
                # metric_f1_o(result['prediction_o'], y['gold_tree'], y['depth'], lens=True, nonterminal=True)
            else:
                metric_f1(result['prediction'], y['gold_tree'], lens=True, nonterminal=True)
            if left_binarization:
                metric_f1_left(result['prediction'], y['gold_tree_left'], y['depth_left'], lens=True, nonterminal=True)
            if right_binarization:
                metric_f1_right(result['prediction'], y['gold_tree_right'], y['depth_right'], lens=True, nonterminal=True)

            self.pf_sum = self.pf_sum + torch.sum(result['depth'], dim=0).detach().cpu()
            metric_ll(result['partition'], x['seq_len'])
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])

        return metric_f1, metric_uas, metric_ll, metric_f1_left, metric_f1_right
        # if not eval_dep:
        #     return metric_f1, metric_ll
        # else:
        #     return metric_f1, metric_uas, metric_ll
