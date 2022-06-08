# -*- coding: utf-8 -*-
import math
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
        heatmap_save_flag = True
        heatmap_dir = os.path.join(self.args.save_dir, 'heatmap')
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir, exist_ok=True)

        for x, _ in t:
            if not hasattr(train_arg, 'warmup_epoch') and hasattr(train_arg, 'warmup'):
                if self.iter >= train_arg.warmup:
                    self.partition = True

            self.optimizer.zero_grad()
            
            if self.partition \
                and hasattr(train_arg, 'soft_loss_target') \
                and hasattr(train_arg, 'soft_loss_mode'):
                # Soft gradients
                loss, z_l = self.model.loss(x, partition=self.partition, soft=True)
                if hasattr(train_arg, 'dambda_warmup') and train_arg.dambda_warmup:
                    # self.dambda = self.model.entropy_rules(probs=True, reduce='mean')

                    ent = self.model.entropy_rules(probs=True, reduce='mean')
                    factor = min(1, max(0, (self.iter-self.num_batch*train_arg.warmup_start)/(self.num_batch*train_arg.warmup_iter)))
                    self.dambda = ent + factor * (1-ent)

                    # self.dambda = self.model.entropy_rules(probs=True).mean()

                    # ent = self.model.entropy_rules(probs=True).mean()
                    # factor = min(1, max(0, (self.iter-20000)/70000))
                    # factor = min(1, self.iter/self.total_iter)
                    # self.dambda = ent + factor * (1-ent)
                else:
                    self.dambda = 1
                records = self.model.soft_backward(
                    loss, z_l, self.optimizer,
                    dambda=self.dambda,
                    target=train_arg.soft_loss_target,
                    mode=train_arg.soft_loss_mode
                )
            else:
                # Hard gradients
                loss = self.model.loss(x, partition=self.partition)
                loss.backward()
                records = None

            if hasattr(train_arg, 'heatmap') and train_arg.heatmap and heatmap_save_flag:
                self.model.save_rule_heatmap(dirname=heatmap_dir, filename=f'rule_dist_{self.iter}.png')
                heatmap_save_flag = False
            
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            # writer
            self.total_loss += loss.item()
            self.total_len += x['seq_len'].max().double()

            if hasattr(self.model, 'pf'):
                self.pf = self.pf + self.model.pf.detach().cpu().tolist() if self.model.pf.numel() != 1 else [self.model.pf.detach().cpu().tolist()]
            if self.iter != 0 and self.iter % 500 == 0:
                self.writer.add_scalar('train/loss', self.total_loss/500, self.iter)
                # self.writer.add_scalar('train/lambda', torch.cat(list(self.dambda.values())).mean(), self.iter)
                self.writer.add_scalar('train/lambda', self.dambda, self.iter)
                self.writer.add_scalar('train/length', self.total_len/500, self.iter)
                self.total_loss = 0
                self.total_len = 0
                if hasattr(self.model, 'pf'):
                    self.writer.add_histogram('train/partition_number', self.model.pf.detach().cpu(), self.iter)
                    self.pf = []
                for k, v in self.model.rules.items():
                    if k == 'kl':
                        continue
                    self.writer.add_histogram(f'train/{k}_prob', v.detach().cpu(), self.iter)
                    self.writer.add_histogram(f'train/{k}_grad', v.grad.detach().cpu(), self.iter)
                if records is not None:
                    for k, v in records.items():
                        self.writer.add_histogram(f'train/{k}', v.detach().cpu(), self.iter)
                ent = self.model.entropy_rules()
                self.writer.add_histogram(f'train/entropy', ent.detach().cpu(), self.iter)
                    
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
