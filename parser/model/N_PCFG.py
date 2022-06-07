import torch
import torch.nn as nn
import torch.distributions as dist
from parser.modules.res import ResLayer

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

import matplotlib.pyplot as plt
import math

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

    def save_rule_heatmap(self):
        plt.rcParams['figure.figsize'] = (64, 48)
        dfs = [r.clone().detach().cpu().numpy() for r in self.rules['rule'][0]]
        vmin = self.rules['rule'][0].min()
        vmax = self.rules['rule'][0].max()

        fig, axs = plt.subplots(nrows=5, ncols=6)
        for df, ax in zip(dfs, axs.flat):
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
        # fig.colorbar(pc)
        plt.savefig('tmp_rules_prop_hardBCL.png')

    def get_max_entropy(self):
        return 2 * math.log(self.NT_T)
    
    def max_entropy(self, num):
        return math.log(num)

    def entropy_root(self, batch=False, probs=False):
        root = self.rules['root']
        if batch:
            b = root.shape[0]
            ent = root.new_zeros((b, ))
            for i in range(b):
                ent[i] = dist.categorical.Categorical(logits=root[i])
        else:
            root = root[0]
            ent = dist.categorical.Categorical(logits=root).entropy().unsqueeze(-1)
        if probs:
            emax = self.max_entropy(root.shape[-1])
            ent = (emax - ent) / emax
        return ent

    def entropy_rules(self, batch=False, probs=False):
        rule = self.rules['rule'].reshape(self.rules['rule'].shape[0], self.NT, -1)
        if batch:
            b = rule.shape[0]
            ent = rule.new_zeros((b, self.NT))
            for i in range(b):
                for j in range(self.NT):
                    ent[i, j] = dist.categorical.Categorical(logits=rule[i, j]).entropy()
        else:
            rule = rule[0]
            ent = rule.new_zeros((self.NT, ))
            for i, r in enumerate(rule):
                ent[i] = dist.categorical.Categorical(logits=r).entropy()
        if probs:
            emax = self.max_entropy(rule.shape[-1])
            ent = (emax - ent) / emax
        return ent

    def entropy_terms(self, batch=False, probs=False):
        terms = self.rules['unary']
        if batch:
            b = terms.shape[0]
            ent = terms.new_zeros((b, self.T))
            for i in range(b):
                for j in range(self.T):
                    ent[i, j] = dist.categorical.Categorical(logits=terms[i, j]).entropy()
        else:
            terms = terms[0]
            ent = terms.new_zeros((self.T, ))
            for i, t in enumerate(terms):
                ent[i] = dist.categorical.Categorical(logits=t).entropy()
        if probs:
            emax = self.max_entropy(terms.shape[-1])
            ent = (emax - ent) / emax
        return ent

    def get_entropy(self, batch=False, probs=False, reduce='mean'):
        r_ent = self.entropy_root(batch=batch, probs=probs)
        n_ent = self.entropy_rules(batch=batch, probs=probs)
        t_ent = self.entropy_terms(batch=batch, probs=probs)
        
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
        # self.save_rule_heatmap()
        # self.entropy_rules()
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
        # self.save_rule_heatmap()

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