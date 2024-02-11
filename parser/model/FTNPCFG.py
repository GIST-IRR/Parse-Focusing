import torch
import torch.nn as nn
from .PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Root_parameterizer
)
from parser.modules.res import ResLayer

from parser.pcfgs.td_partition_function import TDPartitionFunction
from ..pcfgs.tdpcfg import Fastest_TDPCFG
from torch.distributions.utils import logits_to_probs
from torch.distributions import Bernoulli


mask_bernoulli = Bernoulli(torch.tensor([0.3]))

class Nonterm_parameterizer(nn.Module):
    def __init__(self, dim, NT, T, r, nonterm_emb=None, term_emb=None):
        super(Nonterm_parameterizer, self).__init__()
        self.dim = dim
        self.NT = NT
        self.T = T
        self.r = r

        if nonterm_emb is not None:
            self.nonterm_emb = nonterm_emb
        else:
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.dim))

        if term_emb is not None:
            self.term_emb = term_emb
        else:
            self.term_emb = nn.Parameter(torch.randn(self.T, self.dim))

        self.parent_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )
        self.left_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )
        self.right_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )

        # self.rank_proj = nn.Linear(self.dim, self.r, bias=False)
        self.rank_proj = nn.Parameter(torch.randn(self.dim, self.r))

    def forward(self):
        rule_state_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        # head = self.rank_proj(self.parent_mlp(self.nonterm_emb))
        # left = self.rank_proj(self.left_mlp(rule_state_emb))
        # right = self.rank_proj(self.right_mlp(rule_state_emb))
        head = self.parent_mlp(self.nonterm_emb) @ self.rank_proj
        left = self.left_mlp(rule_state_emb) @ self.rank_proj
        right = self.right_mlp(rule_state_emb) @ self.rank_proj
        head = head.softmax(-1)
        left = left.softmax(-2)
        right = right.softmax(-2)
        return head, left, right

class FTNPCFG(PCFG_module):
    def __init__(self, args):
        super(FTNPCFG, self).__init__()
        self.pcfg = Fastest_TDPCFG()
        self.part = TDPartitionFunction()
        self.args = args

        self.NT = getattr(args, "NT", 4500)
        self.T = getattr(args, "T", 9000)
        self.NT_T = self.NT + self.T
        self.V = getattr(args, "V", 10003)

        self.s_dim = getattr(args, "s_dim", 256)
        self.r = getattr(args, "r_dim", 1000)
        self.word_emb_size = getattr(args, "word_emb_size", 200)

        # root
        self.root = Root_parameterizer(self.s_dim, self.NT)

        # Embeddings
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))

        # terms
        self.terms = Term_parameterizer(
            self.s_dim, self.T, self.V, term_emb=self.term_emb)

        # Nonterms
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, self.r,
            nonterm_emb=self.nonterm_emb, term_emb=self.term_emb
        )

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root()
            roots = roots.expand(b, roots.shape[-1])
            return roots

        def terms():
            term_prob = self.terms()
            term_prob = term_prob[torch.arange(self.T)[None,None], x[:,:,None]]
            return term_prob

        def rules():
            head, left, right = self.nonterms()
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root = roots()
        unary = terms()
        head, left, right = rules()

        return {
            'unary': unary,
            'root': root,
            'head': head,
            'left': left,
            'right': right,
        }
    
    def loss(self, input, partition=False, soft=False, label=False, **kwargs):

        self.rules = self.forward(input)

        result = self.pcfg(
            self.rules, self.rules['unary'],
            lens=input['seq_len'],
            label=label
        )
        return -result['partition'].mean()

    def evaluate(self, input, decode_type, depth=0, label=False, **kwargs):
        rules = self.forward(input)

        if decode_type == 'viterbi':
            assert NotImplementedError
        elif decode_type == 'mbr':
            result = self.pcfg(rules, rules['unary'], lens=input['seq_len'], viterbi=False, mbr=True, label=label)
        else:
            raise NotImplementedError

        return result