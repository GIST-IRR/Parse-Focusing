from re import M
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from parser.modules.attentions import ScaledDotProductAttention
from ..modules.res import ResLayer

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

import matplotlib.pyplot as plt
import math
import os
from typing import Dict


class NeuralPCFG(PCFG_module):
    def __init__(self, args, dataset):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.device = dataset.device
        self.args = args

        self.NT = getattr(args, "NT", 30)
        self.T = getattr(args, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = len(dataset.word_vocab)

        self.s_dim = getattr(args, "s_dim", 256)

        # embedding vectors for symbols
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        # additional embeddings
        self.word_emb = nn.Embedding(self.V, self.s_dim)

        # Term FCN
        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

        # Rule FCN
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)
        # Partition function
        self.mode = args.get("mode")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def save_rule_heatmap(
        self,
        rules=None,
        dirname="heatmap",
        filename="rules_prop.png",
        abs=True,
        local=True,
        symbol=True,
    ):
        if rules is None:
            rules = self.rules["rule"][0]
        plt.rcParams["figure.figsize"] = (70, 50)
        dfs = [r.clone().detach().cpu().numpy() for r in rules]
        # min max in seed
        if local:
            vmin = rules.min()
            vmax = rules.max()
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"local_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # min max in local
        if symbol:
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                vmin = df.min()
                vmax = df.max()
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"symbol_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # absolute min max
        if abs:
            vmin = -100.0
            vmax = 0.0
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"global_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

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

        def roots():
            root_emb = self.root_emb
            # root_emb = F.dropout(self.root_emb, p=0.5, training=self.training)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            roots = roots.expand(b, self.NT)
            return roots

        def terms():
            term_emb = self.term_emb
            # term_emb = F.dropout(self.term_emb, p=0.5, training=self.training)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            # term_prob = self.term_from_unary(input, term_prob)

            # kmeans distribution
            # term_emb = self.term_emb / torch.linalg.norm(self.term_emb, dim=-1, keepdim=True)
            # word_emb = self.word_emb / torch.linalg.norm(self.word_emb, dim=-1, keepdim=True)
            # term_prob = torch.matmul(term_emb, word_emb.T) - 1
            # term_prob = term_prob.log_softmax(-1)
            # term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)

            # word2vec distribution
            # term_prob = self.term_mlp(self.term_emb)
            # term_prob = self.term_mlp2(term_prob)
            # term_prob = term_prob.log_softmax(-1)
            # term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb
            # nonterm_emb = F.dropout(self.nonterm_emb, p=0.5, training=self.training)
            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            rule_prob = rule_prob.unsqueeze(0).expand(b, *rule_prob.shape)
            return rule_prob

        root, unary, rule = roots(), terms(), rules()
        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()
            # Masking backward hook
            # def masking(grad):
            #     b, n = x.shape
            #     indices = x[..., None].expand(-1, -1, self.T).permute(0, 2, 1)
            #     mask = indices.new_zeros(b, self.T, self.V).scatter_(2, indices, 1.)
            #     return mask * grad

            # unary.register_hook(masking)

        self.clear_metrics() # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            # 'kl': torch.tensor(0, device=self.device)
        }

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
        terms = self.term_from_unary(input["word"], self.rules["unary"])
        self.rules["word"] = input["word"]

        # Calculate inside algorithm
        # result = self.pcfg(
        #     self.rules, terms, lens=input["seq_len"]
        # )
        # result = self.pcfg(
        #     self.rules, terms, lens=input["seq_len"], topk=4
        # )
        # pf = self.pcfg(
        #     self.rules, terms, lens=input["seq_len"]
        # )
        # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'])

        # log_cos_term = self.cos_sim_max(self.rules['log_cos_term'])
        # log_cos_nonterm = self.cos_sim_max(self.rules['log_cos_nonterm'])

        if partition:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"], topk=2
            )
            sent = self.pcfg(
                self.rules, terms, lens=input["seq_len"]
            )
            self.pf = self.part(
                self.rules, lens=input["seq_len"], mode=self.mode
            )
            if soft:
                return (-result["partition"]), sent["partition"]
            result["partition"] = result["partition"] - sent["partition"]
        else:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"]
            )

        # # Attention-based weight
        # emb_vec = self.word_encoder(input["word"])  # word embedding
        # emb_vec = emb_vec.mean(1)  # mean pooling
        # q, k = self.w_q(emb_vec), self.w_k(emb_vec)
        # output, attn_vec = self.attn(q, k, result["partition"].unsqueeze(1))

        return -result["partition"]
        # return -result['partition'] + 0.5 * pf['partition']
        # return -output.squeeze(1)

    def evaluate(self, input, decode_type, depth=0, **kwargs):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input["word"], self.rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                self.rules,
                terms,
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
            )
            # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == "mbr":
            result = self.pcfg(
                self.rules,
                terms,
                lens=input["seq_len"],
                viterbi=False,
                mbr=True,
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
