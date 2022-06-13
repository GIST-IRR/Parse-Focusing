import torch
import torch.nn as nn
from parser.model.PCFG_module import PCFG_module
from parser.modules.res import ResLayer

from parser.pcfgs.td_partition_function import TDPartitionFunction
from ..pcfgs.tdpcfg import TDPCFG
from torch.distributions.utils import logits_to_probs

class TNPCFG(PCFG_module):
    def __init__(self, args, dataset):
        super(TNPCFG, self).__init__()
        self.pcfg = TDPCFG()
        self.part = TDPartitionFunction()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        #terms
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim), nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))

        # Partition function
        self.mode = args.mode if hasattr(args, 'mode') else None

    @torch.no_grad()
    def entropy_root(self, batch=False, probs=False, reduce='none'):
        return self._entropy(self.rules['root'], batch=batch, probs=probs, reduce=reduce)

    @torch.no_grad()
    def entropy_rules(self, batch=False, probs=False, reduce='none'):
        head = self.rules['head'][0]
        left = self.rules['left'][0]
        right = self.rules['right'][0]

        head = head[:, None, None, :]
        left = left.unsqueeze(1)
        right = right.unsqueeze(0)
        ents = head.new_zeros(self.NT)
        for i, h in enumerate(head):
            t = (left + right + h).logsumexp(-1).reshape(-1)
            ent = logits_to_probs(t) * t
            ent = -ent.sum()
            ents[i] = ent

        if reduce == 'mean':
            ents = ents.mean(-1)
        elif reduce == 'sum':
            ents = ents.sum(-1)

        if probs:
            emax = 2 * self.max_entropy(self.NT+self.T)
            ents = (emax - ents) / emax

        return ents

    @torch.no_grad()
    def entropy_terms(self, batch=False, probs=False, reduce='none'):
        return self._entropy(self.rules['unary'], batch=batch, probs=probs, reduce=reduce)

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            unary.retain_grad()
            head.retain_grad()
            left.retain_grad()
            right.retain_grad()

        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def loss(self, input, partition=False, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        # Partition function
        if partition:
            self.pf = self.part(self.rules, input['seq_len'], mode=self.mode)
            if soft:
                return -result['partition'].mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        return -result['partition'].mean()

    def evaluate(self, input, decode_type, depth=0, **kwargs):
        rules = self.forward(input)
        terms = self.term_from_unary(input, rules['unary'])

        if decode_type == 'viterbi':
            assert NotImplementedError
        elif decode_type == 'mbr':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.part(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()

        return result