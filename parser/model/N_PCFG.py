import torch
import torch.nn as nn
from parser.modules.res import ResLayer

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

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

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        if partition:
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            if soft:
                return -result['partition'].mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        return -result['partition'].mean()

    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        rules = self.forward(input)
        terms = self.term_from_unary(input, rules['unary'])

        if decode_type == 'viterbi':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.part(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()

        return result