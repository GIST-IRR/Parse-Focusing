# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

import math

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
        for x, _ in t:
            if hasattr(train_arg, 'init_depth') and train_arg.init_depth > 0:
                if train_arg.depth_curriculum == 'linear':
                    depth = train_arg.init_depth - ((train_arg.init_depth - train_arg.min_depth)/train_arg.warmup)*(self.iter-2)
                elif train_arg.depth_curriculum == 'exp':
                    depth = train_arg.init_depth/math.sqrt(self.iter-1)
                elif train_arg.depth_curriculum == 'fix':
                    depth = train_arg.min_depth
                depth = math.ceil(depth)
                depth = max(train_arg.min_depth, depth)
                self.model.update_depth(depth)

            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            # writer
            if hasattr(self.model, 'pf'):
                self.pf = self.pf + self.model.pf.detach().cpu().tolist() if self.model.pf.numel() != 1 else [self.model.pf.detach().cpu().tolist()]
            if self.iter % 100 == 1:
                self.writer.add_scalar('train/depth', self.model.depth, self.iter)
                if hasattr(self.model, 'pf'):
                    self.writer.add_histogram('train/partition_number', self.model.pf.detach().cpu(), self.iter)
                    self.pf = []
            # Check total iteration
            self.iter += 1

        return


    @torch.no_grad()
    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None):
        if model == None:
            model = self.model
        model.eval()
        metric_f1 = UF1()
        if eval_dep:
            metric_uas = UAS()
        metric_ll = LikelihoodMetric()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))
        # rules = [] # debugging
        if not hasattr(self.model, 'depth') or self.model.depth == 0:
            self.model.depth = 30
        self.pf_sum = torch.zeros(self.model.depth-2)
        for x, y in t:
            result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
            if 'depth' in y:
                metric_f1(result['prediction'], y['gold_tree'], y['depth'])
            else:
                metric_f1(result['prediction'], y['gold_tree'])
            self.pf_sum = self.pf_sum + torch.sum(result['depth'], dim=0).detach().cpu()
            metric_ll(result['partition'], x['seq_len'])
            # rules.append(result['rules']) # debugging
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])
        # with open('prop_debug_1.pkl', 'wb') as f:
        #     pickle.dump(rules, f)
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll
