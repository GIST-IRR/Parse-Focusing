import sys

import numpy as np
import torch
import torch.nn as nn

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

import matplotlib.pyplot as plt
import math
import os


class ExternalNPCFG(PCFG_module):
    def __init__(self, args):
        super(ExternalNPCFG, self).__init__()
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

        self.lang = getattr(args, "lang", "english")
        self.factor = getattr(args, "factor", "right")

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

        terms_checkpoint = torch.load(
            f'weights/{self.lang}_xbar_{self.factor}_term_pd.pt')

        # terms_checkpoint = self.lognorm_strict(terms_checkpoint)
        terms_checkpoint = self.lognorm_safe(terms_checkpoint)

        self.terms = torch.nn.Parameter(terms_checkpoint)
        # self.terms.requires_grad_(False)

        nonterms_checkpoint = torch.load(
            f'weights/{self.lang}_xbar_{self.factor}_rule_pd.pt')
        nonterms_shape = nonterms_checkpoint.shape
        nonterms_checkpoint = nonterms_checkpoint.reshape(
            nonterms_shape[0], nonterms_shape[1] * nonterms_shape[2]
        )

        # nonterms_checkpoint = self.lognorm_strict(nonterms_checkpoint)
        nonterms_checkpoint = self.lognorm_safe(nonterms_checkpoint)

        self.nonterms = torch.nn.Parameter(nonterms_checkpoint)
        # self.nonterms.requires_grad_(False)

        root_checkpoint = torch.load(
            f'weights/{self.lang}_xbar_{self.factor}_root_pd.pt')

        # root_checkpoint = self.lognorm_strict(root_checkpoint)
        root_checkpoint = self.lognorm_safe(root_checkpoint)

        self.root = torch.nn.Parameter(root_checkpoint)
        # self.root.requires_grad_(False)

    def lognorm_safe(self, x, eps=1e-9):
        x = x / x.sum(-1, keepdims=True) + eps
        return x.log()

    def lognorm_strict(self, x, eps=-1e9):
        x = (x / x.sum(-1, keepdims=True)).log()
        x = x.where(~x.isinf(), torch.full_like(x, eps))
        return x

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

        # root, unary, rule = roots(), terms(), rules()
        
        # root, unary, rule = \
        #     self.root.log_softmax(-1).expand(b, -1), \
        #     self.terms.log_softmax(-1).expand(b, -1, -1), \
        #     self.nonterms.log_softmax(-1).reshape(
        #         self.NT, self.NT_T, self.NT_T).expand(b, -1, -1, -1)

        root, rule, unary = \
            self.root.expand(b, -1), \
            self.nonterms.reshape(
                self.NT, self.NT_T, self.NT_T).expand(b, -1, -1, -1), \
            self.terms.expand(b, -1, -1)
        
        # root, rule, unary = \
        #     self.lognorm_safe(self.root).expand(b, -1), \
        #     self.lognorm_safe(self.nonterms).reshape(
        #         self.NT, self.NT_T, self.NT_T).expand(b, -1, -1, -1), \
        #     self.lognorm_safe(self.terms).expand(b, -1, -1)
        
        # for gradient conflict by using gradients of rules
        if self.training:
            pass
            # root.retain_grad()
            # rule.retain_grad()
            # unary.retain_grad()

        self.clear_metrics() # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            # 'kl': torch.tensor(0, device=self.device)
        }

    def loss(self, input, partition=False, soft=False):
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
            self.pf = self.part(
                self.rules, lens=input["seq_len"], mode=self.mode
            )
            if soft:
                return (-result["partition"]), self.pf
            result["partition"] = result["partition"] - self.pf
        else:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"],
                dropout=self.dropout
            )

        return -result["partition"]

    def evaluate(self, input, decode_type, depth=0, **kwargs):
        self.rules = self.forward(input)
        # NPCFG have same rules for all sentences
        # We need to calculate rules only once
        b = input["word"].shape[0]
        rules = {k: v.expand(b, *v.shape[1:]) for k, v in self.rules.items()}
        terms = self.term_from_unary(input["word"], rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                terms,
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
                dropout=self.dropout
            )
            # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
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
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        return result
