# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

from utils import depth_from_tree, sort_span, span_to_tree, save_rule_heatmap, tensor_to_heatmap

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
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
                # loss, z_l, log_cos = self.model.loss(x, partition=self.partition, soft=True)
                if hasattr(train_arg, 'dambda_warmup') and train_arg.dambda_warmup:
                    # sigmoid
                    if self.iter < train_arg.warmup_start:
                        self.dambda = 0
                    elif self.iter >= train_arg.warmup_start and self.iter < train_arg.warmup_end:
                        self.dambda = 1 / (1 + math.exp((-self.iter+train_arg.warmup_iter)/(self.num_batch/8)))
                    else:
                        # if not self.optim_flag:
                        #     self.optimizer, self.optimizer_tmp = \
                        #     self.optimizer_tmp, self.optimizer
                        #     self.optim_flag = True
                        self.dambda = 1
                elif hasattr(train_arg, 'dambda_step') and train_arg.dambda_step:
                    bound = train_arg.total_iter * train_arg.dambda_step
                    if self.iter < bound:
                        self.dambda = 0
                    else:
                        # if not self.optim_flag:
                        #     self.optimizer, self.optimizer_tmp = \
                        #     self.optimizer_tmp, self.optimizer
                        #     self.optim_flag = True
                        self.dambda = 1

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
                    self.dambda = 1

                # t_loss = (loss + self.dambda * z_l + (1-self.dambda) * log_cos).mean()
                t_loss = (loss + self.dambda * z_l).mean()
                t_loss.backward()
                loss = loss.mean()
                z_l = z_l.mean()
                # log_cos = log_cos.mean()
                records = None

                # records = self.model.soft_backward(
                #     loss, z_l, self.optimizer,
                #     dambda=self.dambda,
                #     target=train_arg.soft_loss_target,
                #     mode=train_arg.soft_loss_mode
                # )
            else:
                # Hard gradients
                loss = self.model.loss(x, partition=self.partition)
                # loss, log_nonterm, log_term = self.model.loss(x, partition=self.partition)
                # loss, log_nonterm, log_term, cos_nonterm, cos_term = self.model.loss(x, partition=self.partition)
                t_loss = loss.mean()
                # t_loss = (loss + log_nonterm).mean()
                # t_loss = ((1-self.dambda) * loss + self.dambda * log_cos).mean()
                # t_loss = (loss + self.dambda * log_cos).mean()
                # t_loss = (loss + log_nonterm + log_term).mean()
                # t_loss = (loss + self.dambda * log_nonterm + self.dambda * log_term).mean()
                # t_loss = ((1-self.dambda)*loss + self.dambda * log_nonterm + self.dambda * log_term).mean()

                # # Cos sim warmup
                # if self.iter > int(self.total_iter/10):
                #     # t_loss = loss.mean()
                #     t_loss = (loss + self.dambda*(log_nonterm + log_term)).mean()
                #     # t_loss = (loss + self.dambda*(log_nonterm + log_term + cos_nonterm + cos_term)).mean()
                #     self.dambda -= 3 / self.total_iter
                #     self.dambda = max(self.dambda, 0)
                # else:
                #     t_loss = (log_nonterm + log_term).mean()

                # # Gram-Schmidt process
                # t_loss = log_nonterm[:, self.step, :self.step].mean(-1).mean()

                # if t_loss < 1e-1:
                #     self.step += 1

                t_loss.backward()
                loss = loss.mean()
                log_nonterm = log_nonterm.mean()
                log_term = log_term.mean()
                records = None
                # self.dambda = max(1 - 2 * self.iter / self.total_iter, 0)
                
            # if 'prev_rules' not in locals():
            #     pass
            #     # save_rule_heatmap(self.model.rules, dirname='figure', filename=f'rule_gradient_{self.iter}.png', grad=True, root=False, unary=False)
            # else:
            #     diff = {}
            #     for k in self.model.rules.keys():
            #         diff[k] = self.model.rules[k] - prev_rules[k]
            #     save_rule_heatmap(diff, dirname='figure', filename=f'rule_diff_{self.iter}.png', root=False, unary=True)
            # prev_rules = self.model.rules

            if hasattr(train_arg, 'heatmap') and train_arg.heatmap:
                if self.iter % int(self.total_iter/10) == 0:
                    self.model.save_rule_heatmap(dirname=heatmap_dir, filename=f'rule_dist_{self.iter}.png')
            
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            # writer
            self.total_loss += loss.item()
            self.total_len += x['seq_len'].max().double()
            self.total_kl_term += self.model.rules['kl_term'].mean().item()
            self.total_kl_nonterm += self.model.rules['kl_nonterm'].mean().item()
            self.total_cos_term += self.model.rules['cos_term'].mean().item()
            self.total_cos_nonterm += self.model.rules['cos_nonterm'].mean().item()
            self.total_log_cos_term += self.model.rules['log_cos_term'].mean().item()
            self.total_log_cos_nonterm += self.model.rules['log_cos_nonterm'].mean().item()

            if hasattr(self.model, 'pf'):
                self.pf = self.pf + self.model.pf.detach().cpu().tolist() if self.model.pf.numel() != 1 else [self.model.pf.detach().cpu().tolist()]
            if self.iter != 0 and self.iter % 500 == 0:
                self.writer.add_scalar('train/loss', self.total_loss/500, self.iter)
                self.writer.add_scalar('train/lambda', self.dambda, self.iter)
                self.writer.add_scalar('train/kl_term', self.total_kl_term/500, self.iter)
                self.writer.add_scalar('train/kl_nonterm', self.total_kl_nonterm/500, self.iter)
                self.writer.add_scalar('train/cos_term', self.total_cos_term/500, self.iter)
                self.writer.add_scalar('train/cos_nonterm', self.total_cos_nonterm/500, self.iter)
                self.writer.add_scalar('train/log_cos_term', self.total_log_cos_term/500, self.iter)
                self.writer.add_scalar('train/log_cos_nonterm', self.total_log_cos_nonterm/500, self.iter)
                # self.writer.add_scalar('train/optimizer_flag', self.optim_flag, self.iter)
                # self.writer.add_scalar('train/length', self.total_len/500, self.iter)
                # tensor_to_heatmap(self.model.rules['kl_term'], dirname=heatmap_dir, filename=f'kl_term_{self.iter}.png')
                # tensor_to_heatmap(self.model.rules['kl_nonterm'], dirname=heatmap_dir, filename=f'kl_nonterm_{self.iter}.png')
                # tensor_to_heatmap(self.model.rules['cos_term'], dirname=heatmap_dir, filename=f'cos_term_{self.iter}.png')
                # tensor_to_heatmap(self.model.rules['cos_nonterm'], dirname=heatmap_dir, filename=f'cos_nonterm_{self.iter}.png')
                # tensor_to_heatmap(self.model.rules['log_cos_term'], dirname=heatmap_dir, filename=f'log_cos_term_{self.iter}.png')
                # tensor_to_heatmap(self.model.rules['log_cos_nonterm'], dirname=heatmap_dir, filename=f'log_cos_nonterm_{self.iter}.png')
                # self.writer.add_figure('train/cos_term', tensor_to_heatmap(self.model.rules['cos_term']), self.iter)
                # self.writer.add_figure('train/cos_nonterm', tensor_to_heatmap(self.model.rules['cos_nonterm']), self.iter)
                # self.writer.add_figure('train/log_cos_term', tensor_to_heatmap(self.model.rules['log_cos_term']), self.iter)
                # self.writer.add_figure('train/log_cos_nonterm', tensor_to_heatmap(self.model.rules['log_cos_nonterm']), self.iter)

                self.writer.add_scalar('train/rule_entropy', self.model.entropy_rules(probs=True, reduce='mean'), self.iter)
                self.total_loss = 0
                self.total_len = 0
                self.total_kl_term = 0
                self.total_kl_nonterm = 0
                self.total_cos_term = 0
                self.total_cos_nonterm = 0
                self.total_log_cos_term = 0
                self.total_log_cos_nonterm = 0
                if hasattr(self.model, 'pf'):
                    self.writer.add_histogram('train/partition_number', self.model.pf.detach().cpu(), self.iter)
                    self.pf = []
                for k, v in self.model.rules.items():
                    if k in ['kl', 'kl_term', 'kl_nonterm', 'cos_term', 'cos_nonterm', 'log_cos_term', 'log_cos_nonterm']:
                        continue
                    self.writer.add_histogram(f'train/{k}_prob', v.detach().cpu(), self.iter)
                    if v.grad is not None:
                        self.writer.add_histogram(f'train/{k}_grad', v.grad.detach().cpu(), self.iter)
                if records is not None:
                    for k, v in records.items():
                        self.writer.add_histogram(f'train/{k}', v.detach().cpu(), self.iter)
                ent = self.model.entropy_rules()
                self.writer.add_histogram(f'train/entropy', ent.detach().cpu(), self.iter)
                # save_rule_heatmap(self.model.rules, dirname='figure', filename=f'rule_{self.iter}.png', root=False, rule=False)
                # diff = {}
                # for k in self.model.rules.keys():
                #     diff[k] = self.model.rules[k] - prev_rules[k]
                # save_rule_heatmap(diff, dirname='figure', filename=f'rule_diff_{self.iter}.png', root=False, unary=True)
                    
            # prev_rules = self.model.rules
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

            result['prediction'] = sort_span(result['prediction'])
            predicted_trees = [span_to_tree(r) for r in result['prediction']]
            # for tree in predicted_trees:
            #     for i, pos in enumerate(tree.treepositions('leaves')):
            #         tree[pos] = y['pos'][i]
            s_depth = [depth_from_tree(t) for t in predicted_trees]
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
                if len(result['prediction'][0][0]) >= 3:
                    metric_f1(result['prediction'], y['gold_tree'], y['depth'], lens=True, nonterminal=True)
                else:
                    metric_f1(result['prediction'], y['gold_tree'], y['depth'], lens=True)
            else:
                if len(result['prediction'][0][0]) >= 3:
                    metric_f1(result['prediction'], y['gold_tree'], lens=True, nonterminal=True)
                else:
                    metric_f1(result['prediction'], y['gold_tree'], lens=True)
            if left_binarization:
                if len(result['prediction'][0][0]) >= 3:
                    metric_f1_left(result['prediction'], y['gold_tree_left'], y['depth_left'], lens=True, nonterminal=True)
                else:
                    metric_f1_left(result['prediction'], y['gold_tree_left'], y['depth_left'], lens=True)
            if right_binarization:
                if len(result['prediction'][0][0]) >= 3:
                    metric_f1_right(result['prediction'], y['gold_tree_right'], y['depth_right'], lens=True, nonterminal=True)
                else:
                    metric_f1_right(result['prediction'], y['gold_tree_right'], y['depth_right'], lens=True)

            self.pf_sum = self.pf_sum + torch.sum(result['depth'], dim=0).detach().cpu()
            metric_ll(result['partition'], x['seq_len'])
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])

        return metric_f1, metric_uas, metric_ll, metric_f1_left, metric_f1_right
        # if not eval_dep:
        #     return metric_f1, metric_ll
        # else:
        #     return metric_f1, metric_uas, metric_ll
