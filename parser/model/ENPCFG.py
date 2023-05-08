import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Nonterm_parameterizer,
    Root_parameterizer
)
from ..modules.res import ResLayer

from torch_support.metric import entropy, pairwise_kl_divergence

import matplotlib.pyplot as plt
import math
import os


class Term_parameterizer(nn.Module):
    def __init__(self, dim, T, V, term_emb=None):
        super().__init__()
        self.dim = dim
        self.T = T
        self.V = V

        if term_emb is None:
            self.term_emb = nn.Parameter(torch.randn(self.T, self.dim))
        else:
            self.term_emb = term_emb

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
    def __init__(self, dim, NT, T, temperature=2., nonterm_emb=None) -> None:
        super().__init__()
        self.dim = dim
        self.NT = NT
        self.T = T
        self.NT_T = self.NT + self.T

        self.temperature = temperature

        if nonterm_emb is None:
            self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.dim))
        else:
            self.nonterm_emb = nonterm_emb

        self.rule_mlp = nn.Linear(self.dim, (self.NT_T) ** 2)
        # self.rule_mlp = nn.Sequential(
        #     nn.Linear(self.dim, self.dim),
        #     ResLayer(self.dim, self.dim),
        #     ResLayer(self.dim, self.dim),
        #     # nn.Linear(self.dim, self.V),
        # )

    def forward(self):
        nonterm_prob = self.rule_mlp(self.nonterm_emb)
        # nonterm_prob = (nonterm_prob/self.temperature).log_softmax(-1)
        return nonterm_prob

class Root_parameterizer(nn.Module):
    def __init__(self, dim, NT, root_emb=None):
        super().__init__()
        self.dim = dim
        self.NT = NT

        if root_emb is None:
            self.root_emb = nn.Parameter(torch.randn(1, self.dim))
        else:
            self.root_emb = root_emb

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

class ENPCFG(PCFG_module):
    def __init__(self, args):
        super(ENPCFG, self).__init__()
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

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.terms = Term_parameterizer(
            self.s_dim, self.T, self.V,
            term_emb=self.term_emb
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, self.temperature,
            nonterm_emb=self.nonterm_emb
        )
        self.root = Root_parameterizer(
            self.s_dim, self.NT,
            root_emb=self.root_emb
        )

        self.child_mlp = nn.Linear(self.s_dim*2, self.s_dim)
        self.word_emb = nn.Parameter(torch.randn(self.V, self.s_dim))

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

        # root = self.root()
        # nonterm = self.nonterms()
        # term = self.terms()

        # R_N = root @ self.nonterm_emb.T

        # C = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        # C = torch.cat([
        #     C.unsqueeze(1).repeat(1, self.NT_T, 1),
        #     C.unsqueeze(0).repeat(self.NT_T, 1, 1)
        # ], dim=-1).reshape(self.NT_T**2, -1)
        # C = self.child_mlp(C)

        # N_C = nonterm @ C.T
        # T_w = term @ self.word_emb.T

        # R2N = R_N.log_softmax(-1).expand(b, self.NT)
        
        # N2C = N_C.log_softmax(-1).expand(b, self.NT, self.NT_T**2)
        # N2C = N2C.reshape(b, self.NT, self.NT_T, self.NT_T)

        # C2N = N_C.log_softmax(0)
        # C2N = C2N.reshape(self.NT, self.NT_T, self.NT_T)

        # T2w = T_w.log_softmax(-1).expand(b, self.T, self.V)

        # w2T = T_w.log_softmax(0)

        # root = R2N
        # rule = N2C
        # unary = T2w

        R_N = self.root()
        root = R_N.log_softmax(-1)
        root = root.expand(b, self.NT)

        R_N_norm = torch.linalg.norm(R_N, dim=-1)

        N_C = self.nonterms()
        N2C = N_C.log_softmax(-1) # N -> N+T N+T
        C2N = N_C.log_softmax(0) # N+T N+T -> N

        rule = N2C.reshape(self.NT, self.NT_T, self.NT_T)
        C2N = C2N.reshape(*rule.shape)

        rule = rule.expand(b, *rule.shape)
        N_C_norm = torch.linalg.norm(N_C, dim=-1)

        T_w = self.terms()
        T2w = T_w.log_softmax(-1) # T -> w
        w2T = T_w.log_softmax(0)  # w -> T

        unary = T2w.expand(b, *T_w.shape)
        T_w_norm = torch.linalg.norm(T_w, dim=-1)
        
        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()
            # # Masking backward hook
            # def masking(grad):
            #     # b, n = x.shape
            #     # indices = x[..., None].expand(-1, -1, self.T).permute(0, 2, 1)
            #     # mask = indices.new_zeros(b, self.T, self.V).scatter_(2, indices, 1.)
            #     print("in the hook!")
            #     return grad * 2

            # unary.register_hook(masking)

        self.clear_metrics() # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "w2T": w2T,
            "C2N": C2N,
            # 'kl': torch.tensor(0, device=self.device)
        }, R_N_norm, N_C_norm, T_w_norm

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
        b = input['word'].shape[0]
        # Calculate rule distributions
        self.rules, R_N_norm, N_C_norm, T_w_norm = self.forward(input)
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
            # C2N = self.rules["C2N"].expand(b, *self.rules["C2N"].shape)
            # w2T = self.rules["w2T"].expand(b, *self.rules["w2T"].shape)
            # w2T = self.term_from_unary(
            #     input["word"], w2T, smooth=self.smooth
            # )

            # Gumbel-max trick
            # C2N_mask = F.gumbel_softmax(
            #     C2N, hard=True, dim=1
            # )
            # w2T_mask = F.gumbel_softmax(w2T, hard=True)

            # Argmax
            # C2N_mask = F.one_hot(
            #     C2N.argmax(1), num_classes=self.NT
            # ).permute(0, 3, 1, 2)
            # w2T_mask = F.one_hot(w2T.argmax(-1), num_classes=self.T)
            
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"],
                # C2N=C2N_mask,
                # w2T=w2T_mask,
                dropout=self.dropout
            )

        # c2n_ent = entropy(self.rules["C2N"].reshape(self.NT, -1).T)
        # w2t_ent = entropy(w2T)
        # c2n_kl = pairwise_kl_divergence(
        #     self.rules["C2N"].reshape(self.NT, -1).T)
        # w2T_kl = pairwise_kl_divergence(w2T, batch=True)
        # return -result["partition"] \
        #     + w2t_ent.mean().expand(b, 1) \
        #     + c2n_ent.mean().expand(b, 1)
        return -result["partition"] \
            + 0.1 * N_C_norm.var().expand(b, 1) / 2 \
            + 0.1 * T_w_norm.var().expand(b, 1) / 2 \
            # + 0.001 * R_N_norm.mean().expand(b, 1) / 2 \
            # + 0.01 * N_C_norm.mean().expand(b, 1) / 2 \
            # + 0.5 * T_w_norm.mean().expand(b, 1) / 2 \
            # - w2T_kl.mean().expand(b, 1) / 2 \
            # + w2T_kl.var().expand(b, 1) / 2
            # + 0.01 * w2t_ent.mean().expand(b, 1)

    def evaluate(self, input, decode_type, depth=0, **kwargs):
        self.rules, _, _, _ = self.forward(input)
        # NPCFG have same rules for all sentences
        # We need to calculate rules only once
        b = input["word"].shape[0]
        self.rules.pop("w2T")
        self.rules.pop("C2N")

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
