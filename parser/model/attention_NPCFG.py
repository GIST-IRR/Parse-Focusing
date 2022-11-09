from re import M
from stat import S_IFDIR
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from parser.modules.attentions import ScaledDotProductAttention
from ..modules.res import ResLayer
from parser.modules.utils import dim_dropout

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import PCFG_module

import matplotlib.pyplot as plt
import math
import os


class NeuralPCFG(PCFG_module):
    def __init__(self, args, dataset):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.device = dataset.device
        self.args = args

        # number of symbols
        self.NT = getattr(args, "NT", 30)
        self.T = getattr(args, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = len(dataset.word_vocab)

        self.s_dim = getattr(args, "s_dim", 256)
        self.init_dropout = getattr(args, "dropout", 0.0)
        self.apply_dropout = self.init_dropout

        # embedding vectors for symbols
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        # self.word_emb = nn.Parameter(torch.randn(self.V, self.s_dim))

        # rule attention layers
        ## ROOT -> NT
        # self.root_q_res = ResLayer(
        #       self.s_dim, self.s_dim, n_layers=1, activation='tanh'
        # )
        self.root_q = nn.Linear(self.s_dim, self.s_dim)
        # self.nt_k_res = ResLayer(
        #       self.s_dim, self.s_dim, n_layers=1, activation='tanh'
        # )
        self.nt_k = nn.Linear(self.s_dim, self.s_dim)
        self.r_nt_attn = ScaledDotProductAttention(log_softmax=True)

        ## NT -> NT+T NT+T
        self.nt_q_res = ResLayer(
            self.s_dim, self.s_dim, n_layers=1, activation="tanh"
        )
        self.nt_q = nn.Linear(self.s_dim, self.s_dim)

        self.nt_t_k = ResLayer(
            self.s_dim, self.s_dim, n_layers=1, activation="tanh"
        )
        self.children_compress = nn.Sequential(
            nn.Linear(self.s_dim * 2, self.s_dim),
            nn.Tanh(),
        )
        self.children_k = nn.Linear(self.s_dim, self.s_dim)
        self.nt_s_attn = ScaledDotProductAttention(log_softmax=True)

        ## T -> word
        # self.t_q_res = ResLayer(
        #     self.s_dim, self.s_dim, n_layers=1, activation='tanh'
        # )
        self.t_q = nn.Linear(self.s_dim, self.s_dim)
        # self.word_k_res = ResLayer(
        #     self.s_dim, self.s_dim, n_layers=1, activation='tanh'
        # )
        self.word_k = nn.Linear(self.s_dim, self.s_dim)
        # self.t_w_attn = ScaledDotProductAttention(log_softmax=True)

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_dropout(self, rate):
        self.apply_dropout = self.init_dropout * rate

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

    def clear_grammar(self):
        # This function is used when the network is updated
        # Updated network will have different rules
        self.rules = None

    def forward(self, input):
        x = input["word"]
        b, n = x.shape[:2]

        def roots():
            # Attention
            root_q = self.root_q(self.root_emb)
            # root_q = self.root_q_res(self.root_emb)
            # root_q = self.root_q(root_q)

            nt_k = self.nt_k(self.nonterm_emb)
            # nt_k = self.nt_k_res(self.nonterm_emb)
            # nt_k = self.nt_k(nt_k)
            _, roots = self.r_nt_attn(root_q, nt_k, None)
            roots = roots.expand(b, self.NT)
            return roots

        def terms():
            t_q = self.t_q_res(self.term_emb)
            t_q = self.t_q(t_q)

            w_k = self.word_k_res(self.word_emb.weight)
            w_k = self.word_k(w_k)
            t_q = self.t_q(self.term_emb)
            w_k = self.word_k(self.word_emb)
            _, terms = self.t_w_attn(t_q, w_k, None)
            terms = terms.expand(b, *terms.shape)
            return terms

        def rules():
            # Attention
            nt_q = self.nt_q_res(self.nonterm_emb)
            nt_q = self.nt_q(nt_q)

            nt_t = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
            nt_t = self.nt_t_k(nt_t)

            children = torch.cat([
                nt_t.repeat(1, self.NT_T).reshape(self.NT_T**2, self.s_dim),
                nt_t.repeat(self.NT_T, 1)
            ], dim=1)

            children = self.children_compress(children)
            children_k = self.children_k(children)
            _, rules = self.nt_s_attn(nt_q, children_k, None)
            rules = rules.reshape(self.NT, self.NT_T, self.NT_T)
            rules = rules.expand(b, *rules.shape)
            return rules

        root, unary, rule = roots(), terms(), rules()
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
            # 'kl': torch.tensor(0, device=self.device)
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
        # terms = torch.randint(0, self.V, input["word"].shape, device=self.device)
        # terms = self.term_from_unary(terms, self.rules["unary"])
        terms = self.term_from_unary(input["word"], self.rules["unary"])
        self.rules["word"] = input["word"]

        if partition:
            result = self.pcfg(
                self.rules, terms, lens=input["seq_len"], topk=1
            )
            sent = self.pcfg(
                self.rules, terms, lens=input["seq_len"]
            )
            # self.pf = self.part(
            #     self.rules, lens=input["seq_len"], mode=self.mode
            # )
            if soft:
                return (-result["partition"]), sent["partition"]
                # return (-sent["partition"]), self.pf
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
            )
            # result = self.pcfg(self.rules, self.rules['unary'], lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
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
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        return result
