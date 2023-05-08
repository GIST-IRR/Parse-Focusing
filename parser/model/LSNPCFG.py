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
    def __init__(
            self, dim, T, V,
            term_emb=None, word_emb=None, activation='relu'
        ):

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
            ResLayer(self.dim, self.dim, activation=activation),
            ResLayer(self.dim, self.dim, activation=activation),
            nn.Linear(self.dim, self.V),
        )

        if word_emb is not None:
            self.term_mlp[3].weight = word_emb

    def forward(self):
        term_prob = self.term_mlp(self.term_emb)
        # term_prob = term_prob.log_softmax(-1)
        return term_prob

class Nonterm_parameterizer(nn.Module):
    def __init__(
            self, dim, NT, T,
            temperature=2., nonterm_emb=None, activation='relu'
        ) -> None:
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
    def __init__(
            self, dim, NT,
            root_emb=None, activation='relu'
        ):
        super().__init__()
        self.dim = dim
        self.NT = NT

        if root_emb is None:
            self.root_emb = nn.Parameter(torch.randn(1, self.dim))
        else:
            self.root_emb = root_emb

        self.root_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            ResLayer(self.dim, self.dim, activation=activation),
            ResLayer(self.dim, self.dim, activation=activation),
            nn.Linear(self.dim, self.NT),
        )

    def forward(self):
        root_prob = self.root_mlp(self.root_emb)
        # root_prob = root_prob.log_softmax(-1)
        return root_prob

class LSNPCFG(PCFG_module):
    def __init__(self, args):
        super(LSNPCFG, self).__init__()
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
        self.activation = getattr(args, "activation", "relu")

        self.word_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.embedding = nn.Embedding(self.V, self.s_dim, _weight=self.word_emb)

        self.terms = Term_parameterizer(
            self.s_dim, self.T, self.V,
            term_emb=self.term_emb,
            word_emb=self.word_emb,
            activation=self.activation
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, self.temperature,
            nonterm_emb=self.nonterm_emb
        )
        self.root = Root_parameterizer(
            self.s_dim, self.NT,
            root_emb=self.root_emb,
            activation=self.activation
        )

        # self.neighbor_mlp = nn.Sequential(
        #     nn.Linear(self.s_dim*2, self.T),
        #     nn.Softmax()
        # )
        self.neighbor_mlp = nn.Linear(self.s_dim*2, self.T)

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

        # Root
        R_N = self.root()
        R2N = R_N.log_softmax(-1)

        root = R2N.expand(b, self.NT)
        R_N_norm = torch.linalg.norm(R_N, dim=-1)

        # Rule
        N_C = self.nonterms()
        N2C = N_C.log_softmax(-1) # N -> N+T N+T
        C2N = N_C.log_softmax(0) # N+T N+T -> N

        N2C = N2C.reshape(self.NT, self.NT_T, self.NT_T)
        C2N = C2N.reshape(*N2C.shape)

        rule = N2C.expand(b, *N2C.shape)
        N_C_norm = torch.linalg.norm(N_C, dim=-1)

        # Unary
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

        self.clear_metrics() # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "w2T": w2T,
            "C2N": C2N,
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

    def loss(self, input,
             partition=False,
             soft=False,
             C2N_mask_ratio=1,
             w2T_mask_ratio=1,
        ):
        x = input['word']
        b = x.shape[0]

        # Calculate rule distributions
        self.rules, R_N_norm, N_C_norm, T_w_norm = self.forward(input)
        terms = self.term_from_unary(
            x, self.rules["unary"],
            smooth=self.smooth
        )
        self.rules["word"] = x

        # Reversed rules
        C2N = self.rules["C2N"].expand(b, *self.rules["C2N"].shape)
        w2T = self.rules["w2T"].expand(b, *self.rules["w2T"].shape)
        w2T = self.term_from_unary(
            input["word"], w2T, smooth=self.smooth
        )

        # Neighbor selection
        left_word = torch.cat([x.new_zeros((b, 1)), x[:, :-1]], dim=-1)
        right_word = torch.cat([x[:, 1:], x.new_zeros((b, 1))], dim=-1)
        left_word = self.embedding(left_word)
        right_word = self.embedding(right_word)
        neighbor_words = torch.cat([left_word, right_word], dim=-1)
        neighbor_words = self.neighbor_mlp(neighbor_words).log_softmax(-1)
        # neighbor_words = self.neighbor_mlp(neighbor_words).softmax(-1)
        neighbor_words_mask = F.one_hot(
            neighbor_words.argmax(-1), num_classes=self.T
        )

        # w2T = (0.5 * w2T.exp() + 0.5 * neighbor_words).log()
        # w2T_w = torch.logaddexp(w2T, neighbor_words) - w2T.new_tensor([2]).log()
        # w2T_w = w2T_w.topk(60, dim=-1)[1]
        # w2T_w_mask = F.one_hot(w2T_w, num_classes=self.T).sum(-2)

        # None
        C2N_mask = None
        # w2T_mask = None

        # Gumbel-max trick
        # C2N_mask = F.gumbel_softmax(
        #     C2N, hard=True, dim=1
        # ).log().clamp(-1e9)
        # w2T_mask = F.gumbel_softmax(w2T, hard=True).log().clamp(-1e9)

        # # Gumbel-softmax trick
        # C2N = F.gumbel_softmax(C2N, dim=1).log()
        # w2T = F.gumbel_softmax(w2T).log()

        # C2N_mask = C2N
        # w2T_mask = w2T
        # w2T_mask = neighbor_words

        # Argmax
        # C2N_mask = F.one_hot(
        #     C2N.argmax(1), num_classes=self.NT
        # ).permute(0, 3, 1, 2)
        
        # # Gradually decreasing mask size
        C2N_mask = F.one_hot(
            C2N.topk(
                # 15,
                max(math.ceil(self.NT/C2N_mask_ratio), int(self.NT/10)),
                dim=1
            )[1],
            num_classes=self.NT
        ).sum(1).permute(0, 3, 1, 2)
        w2T_mask = F.one_hot(
            w2T.topk(
                # 30,
                max(math.ceil(self.T/w2T_mask_ratio), int(self.T/10)),
                dim=-1
            )[1],
            num_classes=self.T
        ).sum(-2)

        # w2T_mask = F.one_hot(w2T.argmax(-1), num_classes=self.T)
        # mask_loss = F.cross_entropy(
        #     neighbor_words.permute(0, 2, 1),
        #     w2T_mask.permute(0, 2, 1).float(),
        #     reduction='none'
        # )

        # C2N_mask = C2N.new_full(C2N.shape, 1/self.NT)
        # w2T_mask = w2T.new_full(w2T.shape, 1/self.T)

        # C2N_mask = torch.cat([
        #     w2T_mask.new_ones(b, self.NT), w2T_mask.sum(1)
        # ], dim=-1)
        # C2N_mask = C2N_mask.unsqueeze(1).expand(b, self.NT_T, self.NT_T) \
        #     + C2N_mask.unsqueeze(2).expand(b, self.NT_T, self.NT_T)
        # C2N_mask = C2N_mask.unsqueeze(1).expand(b, self.NT, self.NT_T, self.NT_T)

        # tmp_mask = (w2T.argmax(-1) + 30)
        # C2N_mask = F.one_hot(
        #     (w2T.argmax(-1)+30)[:, :-1] * self.NT_T \
        #         + (w2T.argmax(-1)+30)[:, 1:],
        #     num_classes=self.NT_T**2
        # ).sum(1).reshape(b, self.NT_T, self.NT_T).unsqueeze(1).expand(-1, self.NT, -1, -1)
        # C2N_mask[:, :, :30, :30] = 1

        # w2T_mask = (w2T_mask + neighbor_words_mask).log().clamp(-1e9)
        # w2T_w_mask = w2T_w_mask.log().clamp(-1e9)

        # C2N_mask = C2N_mask.log().clamp(-1e9)
        w2T_w_mask = w2T_mask.log().clamp(-1e9)

        # w2T_w_mask = None
        # neighbor_words_mask.log().clamp(-1e9)
        
        result = self.pcfg(
            self.rules, terms, lens=input["seq_len"],
            C2N=C2N_mask,
            w2T=w2T_w_mask,
            dropout=self.dropout
        )

        # result_neighbor = self.pcfg(
        #     self.rules, terms, lens=input["seq_len"],
        #     C2N=C2N_mask,
        #     w2T=neighbor_words_mask,
        #     dropout=self.dropout
        # )

        # result['partition'] = torch.logaddexp(
        #     result['partition'],  result_neighbor['partition']
        # )

        # Entropy
        # c2n_ent = entropy(self.rules["C2N"].reshape(self.NT, -1).T)
        # w2t_ent = entropy(w2T)

        # c2n_kl = pairwise_kl_divergence(
        #     self.rules["C2N"].reshape(self.NT, -1).T)
        # w2T_kl = pairwise_kl_divergence(w2T, batch=True)
        # return -result["partition"] \
        #     + w2t_ent.mean().expand(b, 1) \
        #     + c2n_ent.mean().expand(b, 1)
        return -result["partition"] \
            # + N_C_norm.var().expand(b, 1)\
            # + T_w_norm.var().expand(b, 1)\
            # + mask_loss.mean(-1)
            # + w2t_ent.mean(-1) \
            # + c2n_ent.mean().expand(b, 1) \
            # - w2T_kl.mean().expand(b, 1) \
            # + w2T_kl.var().expand(b, 1)
            # + 0.001 * R_N_norm.mean().expand(b, 1) / 2 \
            # + 0.01 * N_C_norm.mean().expand(b, 1) / 2 \
            # + 0.5 * T_w_norm.mean().expand(b, 1) / 2 \

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
