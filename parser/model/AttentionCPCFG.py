import torch
import torch.nn as nn
import torch.nn.functional as F
from parser.model.PCFG_module import PCFG_module
from parser.modules.res import ResLayer
from parser.modules.attentions import ScaledDotProductAttention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG


class Root_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, NT) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.NT = NT
        
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.root_mlp = nn.Sequential(
            # nn.Linear(self.s_dim + self.z_dim, self.s_dim),
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

    def forward(self, z):
        b = z.shape[0]
        root_emb = self.root_emb.expand(b, self.s_dim)
        # root_emb = torch.cat([root_emb, z], -1)
        # root_prob = self.root_mlp(root_emb).log_softmax(-1)
        # root_emb = root_emb * z

        # root_prob = self.root_mlp(root_emb).log_softmax(-1)
        root_prob = self.root_mlp(root_emb)
        # root_prob = torch.matmul(root_prob, z).log_softmax(-1)
        return root_prob

class Term_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, T, V) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.T = T
        self.V = V

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))

        self.term_mlp = nn.Sequential(
            # nn.Linear(self.s_dim + self.z_dim, self.s_dim),
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

    def forward(self, z):
        b = z.shape[0]
        term_emb = self.term_emb.unsqueeze(0).expand(b, -1, -1)
        # z_expand = z.unsqueeze(1).expand(-1, self.T, -1)
        # term_emb = torch.cat([term_emb, z_expand], -1)
        # term_emb = term_emb * z.unsqueeze(1)

        # term_prob = self.term_mlp(term_emb).log_softmax(-1)
        term_prob = self.term_mlp(term_emb)
        return term_prob

class Nonterm_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, NT, T) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))

        # self.nonterm_mlp = nn.Linear(self.s_dim + self.z_dim, (self.NT_T) ** 2)
        self.nonterm_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)

    def forward(self, z):
        b = z.shape[0]
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
            b, self.NT, self.s_dim
        )
        # z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
        # nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
        # nonterm_emb = nonterm_emb * z.unsqueeze(1)

        # rule_prob = self.nonterm_mlp(nonterm_emb).log_softmax(-1)
        rule_prob = self.nonterm_mlp(nonterm_emb)
        # rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
        return rule_prob

class AttentionLayer(nn.Module):
    def __init__(
        self, V, w_dim, h_dim, s_dim, NT, T,
        num_head=8
    ) -> None:
        super().__init__()
        self.V = V
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.NT = NT
        self.T = T
        self.NT_T = NT + T

        self.w_emb = nn.Embedding(self.V, self.w_dim)
        self.q_nn = nn.Linear(self.w_dim, self.h_dim)
        self.k_nn = nn.Linear(self.w_dim, self.h_dim)
        self.v_nn = nn.Linear(self.w_dim, self.h_dim)
        self.attn = ScaledDotProductAttention()

        # self.fnn = nn.Sequential(
        #     nn.Linear(self.h_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.s_dim),
        # )
        self.root_fnn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.NT),
        )
        self.nonterm_fnn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.NT_T**2),
        )
        self.term_fnn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.V),
        )
    
    def forward(self, x):
        w = self.w_emb(x)
        q = self.q_nn(w)
        k = self.k_nn(w)
        v = self.v_nn(w)

        output, _ = self.attn(q, k, v)
        output = output.mean(1)

        # output = self.fnn(output)
        root_z = self.root_fnn(output)
        nonterm_z = self.nonterm_fnn(output)
        term_z = self.term_fnn(output)

        return root_z, nonterm_z, term_z

class EncodingLayer(nn.Module):
    def __init__(
        self, V, w_dim, NT, T, dim_feedforward=2048, num_heads=1
    ) -> None:
        super().__init__()
        self.V = V
        self.w_dim = w_dim
        NT_T = NT + T

        self.w_emb = nn.Embedding(self.V, self.w_dim)
        self.self_attn = nn.modules.activation.MultiheadAttention(
            self.w_dim, num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(self.w_dim)

        self.root = nn.Sequential(
            nn.Linear(self.w_dim, dim_feedforward),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(dim_feedforward, NT),
        )
        self.root_norm = nn.LayerNorm(NT)

        self.rule = nn.Sequential(
            nn.Linear(self.w_dim, dim_feedforward),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(dim_feedforward, NT_T**2),
        )
        self.rule_norm = nn.LayerNorm(NT_T**2)

        self.term = nn.Sequential(
            nn.Linear(self.w_dim, dim_feedforward),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(dim_feedforward, self.V),
        )
        self.term_norm = nn.LayerNorm(self.V)
    
    def forward(self, x):
        x = self.w_emb(x)
        attn_output = self.self_attn(x, x, x)[0]
        # attn_output = self.dropout(attn_output)
        x = self.norm(x + attn_output)

        root = self.root_norm(self.root(x)).mean(dim=1, keepdim=True)
        rule = self.rule_norm(self.rule(x)).mean(dim=1, keepdim=True)
        term = self.term_norm(self.term(x)).mean(dim=1, keepdim=True)

        return root, rule, term

class AttentionCPCFG(PCFG_module):
    def __init__(self, args):
        super(AttentionCPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.NT_T = self.NT + self.T
        self.V = args.V

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.z_dim, self.NT, self.T
        )
        self.terms = Term_parameterizer(
            self.s_dim, self.z_dim, self.T, self.V
        )
        self.root = Root_parameterizer(
            self.s_dim, self.z_dim, self.NT
        )

        # self.attn = AttentionLayer(
        #     self.V, self.w_dim, self.h_dim, self.s_dim, self.NT, self.T
        # )

        self.attn = EncodingLayer(
            self.V, self.w_dim, self.NT, self.T, num_heads=1
        )

        # Partition function
        self.mode = getattr(args, "mode", None)
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def withoutTerm_parameters(self):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != "terms":
                yield param

    def rules_similarity(self, rule=None, unary=None):
        if rule is None:
            rule = self.rules["rule"]
        if unary is None:
            unary = self.rules["unary"]

        b = rule.shape[0]
        
        tkl = self.kl_div(unary) # KLD for terminal
        nkl = self.kl_div(rule) # KLD for nonterminal
        tcs = self.cos_sim(unary) # cos sim for terminal
        ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1)
        ) # cos sim for nonterminal
        log_tcs = self.cos_sim(unary, log=True) # log cos sim for terminal
        log_ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1), log=True
        ) # log cos sim for nonterminal
        
        return {
            "kl_term": tkl,
            "kl_nonterm": nkl,
            "cos_term": tcs,
            "cos_nonterm": ncs,
            "log_cos_term": log_tcs,
            "log_cos_nonterm": log_ncs
        }

    @property
    def metrics(self):
        if getattr(self, "_metrics", None) is None:
            self._metrics = self.rules_similarity()
        return self._metrics

    def clear_metrics(self):
        self._metrics = None

    def forward(self, input, evaluating=False):
        x = input["word"]
        b, n = x.shape[:2]

        root_z, rule_z, term_z = self.attn(x)

        root, rule, unary = self.root(root_z), self.nonterms(rule_z), self.terms(term_z)

        root = (root * root_z.squeeze(1)).log_softmax(-1)
        rule = (rule * rule_z).log_softmax(-1)
        unary = (unary * term_z).log_softmax(-1)

        rule = rule.reshape(b, self.NT, self.NT_T, self.NT_T)

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            # unary.retain_grad()
            rule.retain_grad()

        return {
            "unary": unary,
            "root": root,
            "rule": rule
        }

    def loss(self, input, partition=False, max_depth=0, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input["word"], self.rules["unary"])

        result = self.pcfg(self.rules, terms, lens=input["seq_len"])
        # Partition function
        if partition:
            self.pf = self.part(self.rules, lens=input["seq_len"], mode=self.mode)
            # Renormalization
            if soft:
                return (-result["partition"] + self.rules["kl"]).mean(), self.pf.mean()
            result["partition"] = result["partition"] - self.pf
        # depth-conditioned inside algorithm
        return (-result["partition"]).mean()

    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        rules = self.forward(input, evaluating=True)
        terms = self.term_from_unary(input["word"], rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                terms,
                lens=input["seq_len"],
                viterbi=True,
                mbr=False
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                terms,
                lens=input["seq_len"],
                viterbi=False,
                mbr=True
            )
        else:
            raise NotImplementedError

        if depth > 0:
            result["depth"] = self.part(
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        if "kl" in rules:
            result["partition"] -= rules["kl"]
        return result
