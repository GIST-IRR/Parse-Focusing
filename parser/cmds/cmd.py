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
        total_loss = 0
        for x, _ in t:
            if self.model.mode is not None:
                if self.iter >= train_arg.warmup:
                    self.model.update_switch(True)

            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            # writer
            total_loss += loss.item()
            if hasattr(self.model, 'pf'):
                self.pf = self.pf + self.model.pf.detach().cpu().tolist() if self.model.pf.numel() != 1 else [self.model.pf.detach().cpu().tolist()]
            if self.iter != 0 and self.iter % 100 == 0:
                self.writer.add_scalar('train/depth', self.model.depth, self.iter)
                self.writer.add_scalar('train/loss', total_loss/100, self.iter)
                total_loss = 0
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

        depth = self.args.test.depth if hasattr(self.args.test, 'depth') else 0

        self.pf_sum = torch.zeros(depth + 3)
        self.span_depth = {}
        for x, y in t:
            result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep, depth=depth)
            
            s_depth = [depth_from_span(r) for r in result['prediction']]
            for d in s_depth:
                if d in self.span_depth:
                    self.span_depth[d] += 1
                else:
                    self.span_depth[d] = 1

            if 'depth' in y:
                metric_f1(result['prediction'], y['gold_tree'], y['depth'], lens=True, nonterminal=True)
            else:
                metric_f1(result['prediction'], y['gold_tree'], lens=True, nonterminal=True)
            self.pf_sum = self.pf_sum + torch.sum(result['depth'], dim=0).detach().cpu()
            metric_ll(result['partition'], x['seq_len'])
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])
        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll
