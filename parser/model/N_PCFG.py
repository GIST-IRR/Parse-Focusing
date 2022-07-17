from argparse import ArgumentError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from parser.modules.res import ResLayer

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

import matplotlib.pyplot as plt
import math
import os

class NeuralPCFG(PCFG_module):
    def __init__(self, args, dataset):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.device = dataset.device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)
        # Partition function
        self.mode = args.mode if hasattr(args, 'mode') else None

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def save_rule_heatmap(self, rules=None, dirname='heatmap', filename='rules_prop.png', abs=True, local=True, symbol=True):
        if rules is None:
            rules = self.rules['rule'][0]
        plt.rcParams['figure.figsize'] = (70, 50)
        dfs = [r.clone().detach().cpu().numpy() for r in rules]
        # min max in seed
        if local:
            vmin = rules.min()
            vmax = rules.max()
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f'local_{filename}')
            plt.savefig(path, bbox_inches='tight')
            plt.close()

        # min max in local
        if symbol:
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                vmin = df.min()
                vmax = df.max()
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f'symbol_{filename}')
            plt.savefig(path, bbox_inches='tight')
            plt.close()

        # absolute min max
        if abs:
            vmin = -100.0
            vmax = 0.0
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f'global_{filename}')
            plt.savefig(path, bbox_inches='tight')
            plt.close()

    
    def entropy_root(self, batch=False, probs=False, reduce='none'):
        return self._entropy(self.rules['root'], batch=batch, probs=probs, reduce=reduce)

    def entropy_rules(self, batch=False, probs=False, reduce='none'):
        return self._entropy(self.rules['rule'], batch=batch, probs=probs, reduce=reduce)

    def entropy_terms(self, batch=False, probs=False, reduce='none'):
        return self._entropy(self.rules['unary'], batch=batch, probs=probs, reduce=reduce)

    def get_entropy(self, batch=False, probs=False, reduce='mean'):
        r_ent = self.entropy_root(batch=batch, probs=probs, reduce=reduce)
        n_ent = self.entropy_rules(batch=batch, probs=probs, reduce=reduce)
        t_ent = self.entropy_terms(batch=batch, probs=probs, reduce=reduce)
        
        # ent_prob = torch.cat([r_ent, n_ent, t_ent])
        # ent_prob = ent_prob.mean()
        if reduce == 'none':
            ent_prob = {
                'root': r_ent,
                'rule': n_ent,
                'unary': t_ent
            }
        elif reduce == 'mean':
            ent_prob = torch.cat([r_ent, n_ent, t_ent]).mean()
        return ent_prob

    def forward(self, input):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            return term_prob

        def rules():
            rule_prob = self.rule_mlp(self.nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            return rule_prob.unsqueeze(0).expand(b, *rule_prob.shape).contiguous()

        root, unary, rule = roots(), terms(), rules()
        
        # KLD for terminal
        tkl = unary.new_zeros(b, self.T, self.T)
        for i in range(self.T):
            u = unary[:, i:i+1, :].expand(-1, self.T, -1)
            kl_score = F.kl_div(u, unary, log_target=True, reduction='none')
            kl_score = kl_score.sum(-1)
            tkl[:, i] = kl_score
        # reverse ratio of kl score
        mask = tkl.new_ones(tkl.shape[1:]).fill_diagonal_(0)
        weight = 1 - (tkl / tkl.sum((1, 2), keepdims=True))
        weight = (mask * weight).detach()
        tkl = (weight * tkl).mean((1, 2))
        tkl = tkl.mean()

        # KLD for nonterminal
        nkl = unary.new_zeros(b, self.T, self.T)
        for i in range(self.T):
            u = unary[:, i:i+1, :].expand(-1, self.T, -1)
            kl_score = F.kl_div(u, unary, log_target=True, reduction='none')
            kl_score = kl_score.sum(-1)
            nkl[:, i] = kl_score
        # reverse ratio of kl score
        mask = nkl.new_ones(nkl.shape[1:]).fill_diagonal_(0)
        weight = 1 - (nkl / nkl.sum((1, 2), keepdims=True))
        weight = (mask * weight).detach()
        nkl = (weight * nkl).mean((1, 2))
        nkl = nkl.mean()

        # cos sim for terminal
        tcs = unary.new_zeros(b)
        for i in range(self.T):
            if i == self.T-1:
                continue
            u = unary[:, i:i+1].expand(-1, self.T-i-1, -1)
            o = unary[:, i+1:self.T]
            cosine_score = F.cosine_similarity(u.exp(), o.exp(), dim=2)
            tcs += cosine_score.abs().sum(-1)
        tcs = tcs / (unary.size(1)*(unary.size(1)-1)/2)
        
        # cos sim for nonterminal
        ncs = rule.new_zeros(b)
        for i in range(self.NT):
            if i == self.NT-1:
                continue
            r = rule[:, i:i+1].reshape(b, 1, -1).expand(-1, self.NT-i-1, -1)
            o = rule[:, i+1:self.NT].reshape(b, self.NT-i-1, -1)
            cosine_score = F.cosine_similarity(r.exp(), o.exp(), dim=2)
            ncs += cosine_score.abs().sum(-1)
        ncs = ncs / (rule.size(1)*(rule.size(1)-1)/2)

        # log cos sim for terminal
        log_tcs = unary.new_zeros(b)
        for i in range(self.T):
            if i == self.T-1:
                continue
            u = unary[:, i:i+1].expand(-1, self.T-i-1, -1)
            o = unary[:, i+1:self.T]
            cosine_score = F.cosine_similarity(u, o, dim=2)
            log_tcs += cosine_score.abs().sum(-1)
        log_tcs = log_tcs / (unary.size(1)*(unary.size(1)-1)/2)
        
        # log cos sim for nonterminal
        log_ncs = rule.new_zeros(b)
        for i in range(self.NT):
            if i == self.NT-1:
                continue
            r = rule[:, i:i+1].reshape(b, 1, -1).expand(-1, self.NT-i-1, -1)
            o = rule[:, i+1:self.NT].reshape(b, self.NT-i-1, -1)
            cosine_score = F.cosine_similarity(r, o, dim=2)
            log_ncs += cosine_score.abs().sum(-1)
        log_ncs = log_ncs / (rule.size(1)*(rule.size(1)-1)/2)

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()

        return {'unary': unary,
                'root': root,
                'rule': rule,
                # 'kl': torch.tensor(0, device=self.device)
                'kl_term': tkl,
                'kl_nonterm': nkl,
                'cos_term': tcs,
                'cos_nonterm': ncs,
                'log_cos_term': log_tcs,
                'log_cos_nonterm': log_ncs
                }

    def loss(self, input, partition=False, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        if partition:
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            if soft:
                # return (-result['partition'] + self.rules['kl']).mean(), self.pf.mean()
                return (-result['partition'] + self.rules['kl_nonterm']).mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        # return -result['partition'].mean()
        # return (-result['partition'] + self.rules['kl_term']).mean()
        # return (-result['partition'] + self.rules['kl_term'] + self.rules['kl_nonterm']).mean()
        return (-result['partition'] + self.rules['log_cos_nonterm']).mean()

    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        if decode_type == 'viterbi':
            result = self.pcfg(self.rules, terms, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg(self.rules, terms, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.part(self.rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()

        return result