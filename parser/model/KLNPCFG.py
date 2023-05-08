import numpy as np
import torch
import torch.nn as nn

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from ..modules.res import ResLayer
from .PCFG_module import (
    PCFG_module,
    # Term_parameterizer,
    # Nonterm_parameterizer,
    # Root_parameterizer
)
from torch_support.metric import *

import matplotlib.pyplot as plt
import math
import os


class Term_parameterizer(nn.Module):
    def __init__(self, dim, T, V):
        super().__init__()
        self.dim = dim
        self.T = T
        self.V = V

        self.term_emb = nn.Parameter(torch.randn(self.T, self.dim))

        self.term_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            ResLayer(self.dim, self.dim),
            ResLayer(self.dim, self.dim),
            nn.Linear(self.dim, self.V),
        )

    def forward(self):
        term_prob = self.term_mlp(self.term_emb)
        # term_prob = term_prob.log_softmax(-1)
        return term_prob

class Nonterm_parameterizer(nn.Module):
    def __init__(self, dim, NT, T, temperature=2.) -> None:
        super().__init__()
        self.dim = dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.temperature = temperature

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.dim))

        self.rule_mlp = nn.Linear(self.dim, (self.NT_T) ** 2)

    def forward(self):
        nonterm_prob = self.rule_mlp(self.nonterm_emb)
        # nonterm_prob = (nonterm_prob/self.temperature).log_softmax(-1)
        # nonterm_prob = (nonterm_prob/self.temperature)
        return nonterm_prob

class Root_parameterizer(nn.Module):
    def __init__(self, dim, NT):
        super().__init__()
        self.dim = dim
        self.NT = NT

        self.root_emb = nn.Parameter(torch.randn(1, self.dim))

        self.root_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            ResLayer(self.dim, self.dim),
            ResLayer(self.dim, self.dim),
            nn.Linear(self.dim, self.NT),
        )

    def forward(self):
        root_prob = self.root_mlp(self.root_emb)
        # root_prob = root_prob.log_softmax(-1)
        return root_prob

class KLNPCFG(PCFG_module):
    def __init__(self, args):
        super(KLNPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.args = args

        # number of symbols
        self.NT = getattr(args, "NT", 30)
        self.T = getattr(args, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = getattr(args, "V", 10002)

        self.s_dim = getattr(args, "s_dim", 256)
        self.dropout = getattr(args, "dropout", 0.0)

        self.temperature = getattr(args, "temperature", 1.0)
        self.smooth = getattr(args, "smooth", 0.0)

        self.lamb = getattr(args, "lamb", [0.0, 0.0])

        # self.word_emb = nn.Embedding(self.V, self.s_dim)
        # self.w2t = nn.Linear(self.s_dim, self.s_dim)

        self.terms = Term_parameterizer(
            self.s_dim, self.T, self.V
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, self.temperature
        )

        # self.c2n = nn.Linear(self.s_dim * 2, self.s_dim)

        self.root = Root_parameterizer(
            self.s_dim, self.NT
        )

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

    def withoutTerm_parameters(self):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != "terms":
                yield param

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_dropout(self, rate):
        self.apply_dropout = self.init_dropout * rate

    def entropy(self, key, batch=False, probs=False, reduce="none"):
        assert key == "root" or key == "rule" or key == "unary"
        return self._entropy(
            self.rules[key], batch=batch, probs=probs, reduce=reduce
        )

    def get_entropy(self, batch=False, probs=False, reduce="mean"):
        r_ent = self.entropy("root", batch=batch, probs=probs, reduce=reduce)
        n_ent = self.entropy("rule", batch=batch, probs=probs, reduce=reduce)
        t_ent = self.entropy("unary", batch=batch, probs=probs, reduce=reduce)

        # ent_prob = torch.cat([r_ent, n_ent, t_ent])
        # ent_prob = ent_prob.mean()
        if reduce == "none":
            ent_prob = {"root": r_ent, "rule": n_ent, "unary": t_ent}
        elif reduce == "mean":
            ent_prob = torch.cat([r_ent, n_ent, t_ent]).mean()
        return ent_prob

    def sentence_vectorizer(sent, model):
        sent_vec = []
        numw = 0
        for w in sent:
            try:
                if numw == 0:
                    sent_vec = model.wv[w]
                else:
                    sent_vec = np.add(sent_vec, model.wv[w])
                numw += 1
            except:
                pass
        return np.asarray(sent_vec) / numw

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

    @property
    def rules(self):
        if getattr(self, "_rules", None) is None:
            self._rules = self.forward({"word": torch.zeros([1, 1])})
        return self._rules

    @rules.setter
    def rules(self, rule):
        self._rules = rule

    def forward(self, input):
        x = input["word"]
        b, n = x.shape[:2]

        root, unary, rule = self.root().log_softmax(-1), \
            self.terms().log_softmax(-1), \
            self.nonterms().log_softmax(-1)
        
        rule_kl = pairwise_kl_divergence(rule)
        unary_kl = pairwise_kl_divergence(unary)

        root = root.expand(b, self.NT)
        unary = unary.expand(b, *unary.shape)
        rule = rule.reshape(self.NT, self.NT_T, self.NT_T)
        rule = rule.expand(b, *rule.shape)
        
        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()

        self.clear_metrics() # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "nonterm_cs": rule_kl,
            "term_cs": unary_kl
        }

    def partition_function(self, max_length=200):
        return self.part(
            self.rules, lens=max_length, mode='depth', until_converge=True
        )

    def unique_terms(self, terms):
        b, n = terms.shape
        for t in terms:
            output, inverse, counts = torch.unique(
                t, return_inverse=True, return_counts=True
            )
            duplicated_index = counts.where(counts > 1)

    def loss(self, input, partition=False, soft=False):
        # b = input['word'].shape[0]
        # Calculate rule distributions
        self.rules = self.forward(input)
        terms = self.term_from_unary(
            input["word"], self.rules["unary"],
            smooth=self.smooth
        )
        self.rules["word"] = input["word"]

        if partition:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"], topk=1
            )
            # sent = self.pcfg(
            #     self.rules, terms, lens=input["seq_len"]
            # )
            self.pf = self.part(
                self.rules, lens=input["seq_len"], mode=self.mode
            )
            if soft:
                # return (-result["partition"]), sent["partition"]
                return (-result["partition"]), self.pf
                # return (-sent["partition"]), self.pf
            # result["partition"] = result["partition"] - sent["partition"]
            result["partition"] = result["partition"] - self.pf
        else:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"],
                dropout=self.dropout
            )

        nonterm_cs = self.rules['nonterm_cs'].mean()
        term_cs = self.rules['term_cs'].mean()
        nonterm_kl_var = self.rules['nonterm_cs'].var()
        term_kl_var = self.rules['term_cs'].var()

        return -result["partition"] \
            - self.lamb[0] * nonterm_cs \
            - self.lamb[1] * term_cs \
            + 0.1 * nonterm_kl_var / 2 \
            + 0.1 * term_kl_var / 2

    def evaluate(self, input, decode_type, depth=0, **kwargs):
        self.rules = self.forward(input)
        # NPCFG have same rules for all sentences
        # We need to calculate rules only once
        b = input["word"].shape[0]
        # rules = {k: v.expand(b, *v.shape[1:]) for k, v in self.rules.items()}
        terms = self.term_from_unary(input["word"], self.rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                self.rules,
                terms,
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
                dropout=self.dropout
            )
            # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == "mbr":
            result = self.pcfg(
                self.rules,
                terms,
                lens=input["seq_len"],
                viterbi=False,
                mbr=True,
                dropout=self.dropout
            )
            # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result["depth"] = self.part(
                self.rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        return result
