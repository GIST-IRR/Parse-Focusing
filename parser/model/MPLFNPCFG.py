from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import Faster_PCFG, PCFG
from .PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Nonterm_parameterizer as NT_param,
    Root_parameterizer,
)
from .FTNPCFG import FTNPCFG
from utils import generate_random_span_by_length


class Nonterm_parameterizer(NT_param):
    def __init__(
        self,
        dim,
        NT,
        T,
        nonterm_emb=None,
        term_emb=None,
        activation="relu",
        softmax=True,
        norm=None,
    ) -> None:
        super().__init__(
            dim,
            NT,
            T,
            nonterm_emb=nonterm_emb,
            term_emb=term_emb,
            no_rule_layer=True,
            softmax=softmax,
            norm=norm,
        )
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "tanh":
            activation = nn.Tanh

        self.children_mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            activation(),
            nn.Linear(self.dim, self.dim),
            activation(),
        )

    def forward(self, reshape=False):
        children_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        children_emb = torch.cat(
            [
                children_emb.repeat(1, self.NT_T).reshape(
                    self.NT_T**2, self.dim
                ),
                children_emb.repeat(self.NT_T, 1),
            ],
            dim=1,
        )
        children_emb = self.children_mlp(children_emb)
        rule_prob = self.nonterm_emb @ children_emb.T

        if self.norm is not None:
            rule_prob = self.norm(rule_prob)

        if self.softmax:
            rule_prob = rule_prob.log_softmax(-1)

        if reshape:
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)

        return rule_prob


class MPLFNPCFG(PCFG_module):
    """The model use N-PCFG as pre-parsers"""

    def __init__(self, args):
        super(MPLFNPCFG, self).__init__()
        self.pcfg = Faster_PCFG()
        self.part = PartitionFunction()
        self.args = args

        # number of symbols
        self.NT = getattr(args, "NT", 30)
        self.T = getattr(args, "T", 60)
        self.NT_T = self.NT + self.T
        self.V = getattr(args, "V", 10003)

        self.s_dim = getattr(args, "s_dim", 256)
        self.dropout = getattr(args, "dropout", 0.0)

        self.temperature = getattr(args, "temperature", 1.0)
        self.smooth = getattr(args, "smooth", 0.0)

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.activation = getattr(args, "activation", "relu")
        self.norm = getattr(args, "norm", None)

        self.terms = Term_parameterizer(
            self.s_dim,
            self.T,
            self.V,
            term_emb=self.term_emb,
            activation=self.activation,
            norm=self.norm,
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim,
            self.NT,
            self.T,
            nonterm_emb=self.nonterm_emb,
            term_emb=self.term_emb,
            norm=self.norm,
        )
        # self.nonterms = NT_param(self.s_dim, self.NT, self.T)
        self.root = Root_parameterizer(
            self.s_dim,
            self.NT,
            root_emb=self.root_emb,
            nonterm_emb=self.nonterm_emb,
            activation=self.activation,
            norm=self.norm,
        )

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

        self.parser_type = getattr(args, "parser_type", None)

        self.pretrained_models = getattr(args, "pretrained_models", None)
        self.parse_trees = []
        if self.pretrained_models:
            for model in self.pretrained_models:
                parses = defaultdict(list)
                with open(model, "rb") as f:
                    pts = torch.load(f)
                for t in pts:
                    n_word = t["word"]
                    if "pred_tree" in t.keys():
                        n_tree = [s for s in t["pred_tree"] if s[1] - s[0] > 1]
                    elif "tree" in t.keys():
                        n_tree = [s for s in t["tree"] if s[1] - s[0] > 1]
                    parses[str(n_word)] = n_tree
                self.parse_trees.append(parses)
        # else:
        #     raise NotImplementedError("No pretrained models.")

        # self.tree_mask =

    def withoutTerm_parameters(self):
        for name, param in self.named_parameters():
            module_name = name.split(".")[0]
            if module_name != "terms":
                yield param

    def _initialize(self):
        # Original Method
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        # # Init with constant 0.0009
        # for n, p in self.named_parameters():
        #     n = n.split(".")[0]
        #     if n == "terms":
        #         torch.nn.init.constant_(p, 0.0009)
        #     else:
        #         if p.dim() > 1:
        #             torch.nn.init.xavier_uniform_(p)
        # # Init with mean of each layer
        # for n, p in self.named_parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     if n.split(".")[0] == "terms":
        #         val = p.mean()
        #         torch.nn.init.constant_(p, val)
        # # Init with mean of each layer for whole
        # # This initialization makes the whole rule distribution close to uniform distribution
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     val = p.mean()
        #     torch.nn.init.constant_(p, val)

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
        root = self.root()
        # root = root.expand(b, self.NT)

        # Rule
        rule = self.nonterms()
        rule = rule.reshape(self.NT, self.NT_T, self.NT_T)
        # rule = rule.expand(b, *rule.shape)

        # Unary
        unary = self.terms()
        # unary = unary[torch.arange(self.T)[None, None], x[:, :, None]]

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()

        self.clear_metrics()  # clear metrics becuase we have new rules

        return {
            "root": root,
            "rule": rule,
            "unary": unary,
            # 'kl': torch.tensor(0, device=self.device)
        }

    def partition_function(self, max_length=200):
        return self.part(
            self.rules, lens=max_length, mode="depth", until_converge=True
        )

    def batchify(self, words, rules):
        b, _ = words.shape
        return {
            "root": rules["root"].expand(b, self.NT),
            "rule": rules["rule"].expand(b, *rules["rule"].shape),
            "unary": rules["unary"][
                torch.arange(self.T)[None, None], words[:, :, None]
            ],
        }

    def loss(self, input, parser=True, temp=1, **kwargs):
        words = input["word"]
        # # Sequence permutate randomly
        # assert words is not None
        # # not_used_words = words[:, torch.randperm(words.shape[1])]
        # input["word"] = words[:, torch.randperm(words.shape[1])]
        b, seq_len = words.shape

        # Calculate rule distributions
        self.rules = self.forward(input)
        rules = self.batchify(words, self.rules)

        trees = None
        if "gold_tree" in kwargs.keys() and kwargs["gold_tree"] is not None:
            trees = [[s[:2] for s in t[1:]] for t in kwargs["gold_tree"]]
            trees = words.new_tensor(trees)
        elif self.parser_type == "left_branching":
            trees = words.new_tensor(
                [
                    [[0, seq_len - i] for i in range(seq_len - 1)]
                    for _ in range(b)
                ]
            )
        elif self.parser_type == "right_branching":
            trees = words.new_tensor(
                [[[i, seq_len] for i in range(seq_len - 1)] for _ in range(b)]
            )

        if self.pretrained_models:
            trees = []
            # # Randomly selected trees
            # trees = words.new_tensor(
            #     [generate_random_span_by_length(seq_len) for _ in range(b)]
            # )

            # Tree selected from pre-trained parser
            for parse in self.parse_trees:
                tree = words.new_tensor(
                    [parse.get(str(words[i].tolist())) for i in range(b)]
                )
                trees.append(tree)

            # Calculate span frequency in predicted trees
            trees = torch.stack(trees, dim=1)

            # Original mask
            # tree_mask = trees.new_zeros(
            #     b, trees.shape[1], seq_len + 1, seq_len + 1
            # ).float()
            # for i, t in enumerate(trees):
            #     for j, s in enumerate(t):
            #         for k in s:
            #             tree_mask[i, j, k[0], k[1]] += 1

            # # Random tree weighted mask
            # # trees = trees[:, torch.randint(0, trees.shape[1], (1,))]
            # tree_mask = trees.new_zeros(b, seq_len + 1, seq_len + 1).float()
            # for i, b in enumerate(trees):
            #     selected = torch.randint(0, b.shape[0], (1,))
            #     for j, t in enumerate(b):
            #         weight = 2 if j == selected else 1
            #         for s in t:
            #             tree_mask[i, s[0], s[1]] += weight

            # # Concatenated mask
            trees = trees.reshape(b, -1, 2)

        if trees is not None:
            tree_mask = trees.new_zeros(b, seq_len + 1, seq_len + 1).float()
            for i, t in enumerate(trees):
                for s in t:
                    tree_mask[i, s[0], s[1]] += 1

            # # Softmax mask
            # idx0, idx1 = torch.triu_indices(
            #     seq_len + 1, seq_len + 1, offset=2
            # ).unbind()
            # masked_data = tree_mask[:, idx0, idx1]
            # masked_data = (masked_data / temp).softmax(-1)
            # tree_mask[:, idx0, idx1] = masked_data

            result = self.pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                dropout=self.dropout,
                # tree=trees,
                tree=tree_mask,
                # tree=None,
            )
        else:
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                dropout=self.dropout,
            )

        return -result["partition"]

    def evaluate(self, input, decode_type, depth=0, label=False, **kwargs):
        self.rules = self.forward(input)
        rules = self.batchify(input["word"], self.rules)
        # NPCFG have same rules for all sentences
        # We need to calculate rules only once
        b = input["word"].shape[0]
        lens = input["seq_len"]
        # lens = input['seq_len'].repeat(2)
        # rules = {k: v.expand(b, *v.shape[1:]) for k, v in rules.items()}
        # terms = self.term_from_unary(input["word"], self.rules["unary"])

        if decode_type == "viterbi":
            # Faster_PCFG is not work for viterbi
            result = PCFG()(
                rules,
                rules["unary"],
                lens=lens,
                viterbi=True,
                mbr=False,
                dropout=self.dropout,
                label=label,
            )
            # )
            # result = self.pcfg(
            #     rules,
            #     rules["unary"],
            #     lens=lens,
            #     viterbi=True,
            #     mbr=False,
            #     dropout=self.dropout,
            #     label=label,
            # )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=lens,
                viterbi=False,
                mbr=True,
                dropout=self.dropout,
                label=label,
            )
        else:
            raise NotImplementedError

        if depth > 0:
            result["depth"] = self.part(
                rules, depth, mode="length", depth_output="full"
            )
            result["depth"] = result["depth"].exp()

        return result

    def generate(self):
        self.rules = self.forward({"word": torch.zeros([1, 1])})
        # sampling from root -> NT
        # recursively sampling from NT -> NT or T
        # if all T, then stop
