import torch
import torch.nn as nn
from .PCFG_module import PCFG_module, Term_parameterizer, Root_parameterizer
from parser.modules.res import ResLayer

from parser.pfs.td_partition_function import TDPartitionFunction
from ..pcfgs.tdpcfg import Fastest_TDPCFG
from ..pcfgs.pcfg import PCFG
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
            nn.Linear(self.dim, self.dim), nn.ReLU()
        )
        self.left_mlp = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU())
        self.right_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.ReLU()
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
            self.s_dim, self.T, self.V, term_emb=self.term_emb
        )

        # Nonterms
        self.nonterms = Nonterm_parameterizer(
            self.s_dim,
            self.NT,
            self.T,
            self.r,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
        )

        self._initialize(mode="xavier_normal")

    def compose(self, rules):
        head = rules["head"]
        left = rules["left"]
        right = rules["right"]
        unary = rules["unary"]
        root = rules["root"]

        h = head.shape[0]
        l = left.shape[0]
        r = right.shape[0]

        rule = torch.einsum("ir, jr, kr -> ijk", head, left, right)
        rule = rule.log().reshape(h, l, r)

        return {"unary": unary, "root": root, "rule": rule}

    def forward(self, **kwargs):
        root = self.root()
        unary = self.terms()
        head, left, right = self.nonterms()

        return {
            "unary": unary,
            "root": root,
            "head": head,
            "left": left,
            "right": right,
        }

    def batchify(self, rules, words):
        b = words.shape[0]

        root = rules["root"]
        unary = rules["unary"]

        root = root.expand(b, root.shape[-1])
        unary = unary[torch.arange(self.T)[None, None], words[:, :, None]]

        if len(rules.keys()) == 5:
            head = rules["head"]
            left = rules["left"]
            right = rules["right"]
            head = head.unsqueeze(0).expand(b, *head.shape)
            left = left.unsqueeze(0).expand(b, *left.shape)
            right = right.unsqueeze(0).expand(b, *right.shape)
            return {
                "unary": unary,
                "root": root,
                "head": head,
                "left": left,
                "right": right,
            }

        elif len(rules.keys()) == 3:
            rule = rules["rule"]
            rule = rule.unsqueeze(0).expand(b, *rule.shape)
            return {
                "unary": unary,
                "root": root,
                "rule": rule,
            }

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):

        self.rules = self.forward()
        self.rules = self.batchify(self.rules, input["word"])

        result = self.pcfg(
            self.rules, self.rules["unary"], lens=input["seq_len"], label=label
        )
        return -result["partition"].mean()

    def evaluate(
        self,
        input,
        decode_type="mbr",
        depth=0,
        label=False,
        rule_update=False,
        **kwargs
    ):
        if rule_update:
            need_update = True
        else:
            if hasattr(self, "rules"):
                need_update = False
            else:
                need_update = True

        if need_update:
            self.rules = self.forward()

        if decode_type == "viterbi":
            if not hasattr(self, "viterbi_pcfg"):
                self.viterbi_pcfg = PCFG()
                self.rules = self.compose(self.rules)

        rules = self.batchify(self.rules, input["word"])

        if decode_type == "viterbi":
            result = self.viterbi_pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
                label=label,
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=False,
                mbr=True,
                label=label,
            )
        else:
            raise NotImplementedError

        return result
