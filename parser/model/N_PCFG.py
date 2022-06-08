from argparse import ArgumentError
import torch
import torch.nn as nn
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

    def save_rule_heatmap(self, dirname='heatmap', filename='rules_prop.png'):
        plt.rcParams['figure.figsize'] = (70, 50)
        dfs = [r.clone().detach().cpu().numpy() for r in self.rules['rule'][0]]
        # min max in seed
        vmin = self.rules['rule'][0].min()
        vmax = self.rules['rule'][0].max()
        fig, axes = plt.subplots(nrows=5, ncols=6)
        for df, ax in zip(dfs, axes.flat):
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'local_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.cla()

        # min max in local
        fig, axes = plt.subplots(nrows=5, ncols=6)
        for df, ax in zip(dfs, axes.flat):
            vmin = df.min()
            vmax = df.max()
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'parent_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.cla()

        # absolute min max
        vmin = -100.0
        vmax = 0.0
        fig, axes = plt.subplots(nrows=5, ncols=6)
        for df, ax in zip(dfs, axes.flat):
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'global_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.cla()

    
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

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()
            # unary_full.retain_grad()

        return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': torch.tensor(0, device=self.device)}

    def loss(self, input, partition=False, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        num_t = self.num_trees(input['seq_len'][0])
        num_t = num_t if num_t != 1 else 2

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        if partition:
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            if soft:
                return -result['partition'].mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        return -result['partition'].mean()
        # return -(result['partition']/math.log(num_t)).mean()

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