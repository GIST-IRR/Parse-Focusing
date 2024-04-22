import torch
import torch.nn as nn
import torch.nn.functional as F
from parser.model.PCFG_module import PCFG_module
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from parser.pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG


class Root_parameterizer(nn.Module):
    def __init__(self, s_dim, z_dim, NT) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.NT = NT

        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim + self.z_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

    def forward(self, z):
        b = z.shape[0]
        root_emb = self.root_emb.expand(b, self.s_dim)
        root_emb = torch.cat([root_emb, z], -1)

        root_prob = self.root_mlp(root_emb).log_softmax(-1)
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
            nn.Linear(self.s_dim + self.z_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

    def forward(self, z):
        b = z.shape[0]
        term_emb = self.term_emb.unsqueeze(0).expand(b, -1, -1)
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
        self.nonterm_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)

    def forward(self, z):
        b = z.shape[0]
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
            b, self.NT, self.s_dim
        )
        rule_prob = self.nonterm_mlp(nonterm_emb)
        return rule_prob


class Encoder(nn.Module):
    def __init__(self, V, w_dim, h_dim, z_dim) -> None:
        super().__init__()
        self.V = V
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.enc_emb = nn.Embedding(self.V, self.w_dim)

        self.enc_rnn = nn.LSTM(
            self.w_dim,
            self.h_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        self.enc_out = nn.Linear(self.h_dim * 2, self.z_dim * 2)

    def forward(self, x, len):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, len.cpu(), batch_first=True, enforce_sorted=False
        )
        h_packed, _ = self.enc_rnn(x_packed)
        padding_value = float("-inf")
        output, lengths = pad_packed_sequence(
            h_packed, batch_first=True, padding_value=padding_value
        )
        h = output.max(1)[0]
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar


class CompoundPCFG(PCFG_module):
    def __init__(self, args):
        super(CompoundPCFG, self).__init__()
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
        self.terms = Term_parameterizer(self.s_dim, self.z_dim, self.T, self.V)
        self.root = Root_parameterizer(self.s_dim, self.z_dim, self.NT)

        self.enc = Encoder(self.V, self.w_dim, self.h_dim, self.z_dim)

        # Partition function
        self.mode = getattr(args, "mode", None)
        self._initialize(mode="xavier_uniform")

    def rules_similarity(self, rule=None, unary=None):
        if rule is None:
            rule = self.rules["rule"]
        if unary is None:
            unary = self.rules["unary"]

        b = rule.shape[0]

        tkl = self.kl_div(unary)  # KLD for terminal
        nkl = self.kl_div(rule)  # KLD for nonterminal
        tcs = self.cos_sim(unary)  # cos sim for terminal
        ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1)
        )  # cos sim for nonterminal
        log_tcs = self.cos_sim(unary, log=True)  # log cos sim for terminal
        log_ncs = self.cos_sim(
            rule.reshape(b, self.NT, -1), log=True
        )  # log cos sim for nonterminal

        return {
            "kl_term": tkl,
            "kl_nonterm": nkl,
            "cos_term": tcs,
            "cos_nonterm": ncs,
            "log_cos_term": log_tcs,
            "log_cos_nonterm": log_ncs,
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
        seq_len = input["seq_len"]

        def kl(mean, logvar):
            result = -0.5 * (
                logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1
            )
            return result

        # mean, lvar = enc(x)
        # z = mean

        mean, lvar = self.enc(x, seq_len)
        z = torch.cat([mean, lvar], -1)

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0, 1)
            z = (0.5 * lvar).exp() * z + mean

        root, unary, rule = self.root(z), self.terms(z), self.nonterms(z)

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            # unary.retain_grad()
            rule.retain_grad()

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "kl": kl(mean, lvar).sum(1),
        }

    def loss(self, input, partition=False, max_depth=0, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input["word"], self.rules["unary"])

        result = self.pcfg(self.rules, terms, lens=input["seq_len"])
        return (-result["partition"] + self.rules["kl"]).mean()

    def evaluate(
        self, input, decode_type, depth=0, depth_mode=False, **kwargs
    ):
        rules = self.forward(input, evaluating=True)
        terms = self.term_from_unary(input["word"], rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                rules, terms, lens=input["seq_len"], viterbi=True, mbr=False
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules, terms, lens=input["seq_len"], viterbi=False, mbr=True
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
