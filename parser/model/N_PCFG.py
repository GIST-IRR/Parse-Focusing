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
                # torch.nn.init.orthogonal_(p)
        # for emb in [self.term_emb, self.nonterm_emb]:
        #     torch.nn.init.orthogonal_(emb)
            # cos_sim = F.cosine_similarity(emb[0], emb[1], dim=0)

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
            roots = roots.expand(b, self.NT)
            return roots

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            return term_prob

        def rules():
            rule_prob = self.rule_mlp(self.nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            rule_prob = rule_prob.unsqueeze(0).expand(b, *rule_prob.shape)
            return rule_prob

        # def gram_schmidt(vv):
        #     def projection(u, v):
        #         proj = (v * u).sum() / (u * u).sum() * u
        #         return proj.contiguous()

        #     n, d = vv.shape
        #     uu = vv.new_zeros(vv.shape)
        #     uu[0].copy_(vv[0])
        #     # uu[0].copy_(vv[0] / torch.linalg.norm(vv[0]))
        #     for k in range(1, n):
        #         # vk = vv[k].clone()
        #         uk = vv[k].clone()
        #         # uk = 0
        #         for j in range(0, k):
        #             uk = uk - projection(uu[j].clone(), uk)
        #         uu[k].copy_(uk)
        #         # uu[k].copy_(uk / torch.linalg.norm(uk))
        #     # for k in range(nk):
        #     #     uk = uu[:, k].clone()
        #     #     uu[:, k] = uk / uk.norm()
        #     return uu.contiguous()

        root, unary, rule = roots(), terms(), rules()
        
        # gs_rule = gram_schmidt(rule.reshape(self.NT, -1))
        # nm = torch.linalg.norm(gs_rule[0]) / torch.linalg.norm(gs_rule, dim=-1)
        # gs_rule = nm.unsqueeze(1) * gs_rule
        # gs_rule = gs_rule.reshape(self.NT, self.NT_T, self.NT_T)
        # rule = gs_rule.expand(b, -1, -1, -1)

        # KLD for terminal
        tkl = self.kl_div(unary)
        # reverse ratio of kl score
        # mask = tkl.new_ones(tkl.shape[1:]).fill_diagonal_(0)
        # weight = 1 - (tkl / tkl.sum((1, 2), keepdims=True))
        # weight = (mask * weight).detach()
        # tkl = (weight * tkl).mean((1, 2))
        # tkl = tkl.mean()

        # KLD for nonterminal
        nkl = self.kl_div(rule)
        # reverse ratio of kl score
        # mask = nkl.new_ones(nkl.shape[1:]).fill_diagonal_(0)
        # weight = 1 - (nkl / nkl.sum((1, 2), keepdims=True))
        # weight = (mask * weight).detach()
        # nkl = (weight * nkl).mean((1, 2))
        # nkl = nkl.mean()

        # cos sim for terminal    
        tcs = self.cos_sim(unary)
        
        # cos sim for nonterminal
        ncs = self.cos_sim(rule.reshape(b, self.NT, -1))

        # log cos sim for terminal
        log_tcs = self.cos_sim(unary, log=True)
        
        # log cos sim for nonterminal
        log_ncs = self.cos_sim(rule.reshape(b, self.NT, -1), log=True)

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

        # log cos sim for terminal
        # log_tcs = self.cos_sim(terms, log=True)

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        log_cos_nonterm = self.cos_sim_mean(self.rules['log_cos_nonterm'])
        if partition:
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            if soft:
                return (-result['partition']), self.pf, log_cos_nonterm
                # return (-result['partition'] + self.rules['kl_nonterm']).mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        # return -result['partition'].mean()
        # return (-result['partition'] + self.rules['kl_term']).mean()
        # return (-result['partition'] + self.rules['kl_term'] + self.rules['kl_nonterm']).mean()
        # log_cos_nonterm = self.cos_sim_mean(self.rules['log_cos_nonterm'])
        # return (-result['partition'] + log_cos_nonterm).mean()
        return -result['partition'], log_cos_nonterm

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