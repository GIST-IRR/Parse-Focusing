import numpy as np
import torch

from parser.pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG
from .PCFG_module import (
    PCFG_module,
    Term_parameterizer,
    Nonterm_parameterizer,
    Root_parameterizer,
)


class NeuralPCFG(PCFG_module):
    def __init__(self, args):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
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

        self.terms = Term_parameterizer(self.s_dim, self.T, self.V)
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, self.temperature
        )
        self.root = Root_parameterizer(self.s_dim, self.NT)

        # Partition function
        self.mode = getattr(args, "mode", "length_unary")

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

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

        def roots():
            roots = self.root()
            roots = roots.expand(b, self.NT)
            return roots

        def terms():
            term_prob = self.terms()
            term_prob = term_prob[
                torch.arange(self.T)[None, None], x[:, :, None]
            ]
            # term_prob = term_prob.expand(b, *term_prob.shape)
            return term_prob

        def rules():
            rule_prob = self.nonterms()
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            rule_prob = rule_prob.expand(b, *rule_prob.shape)
            return rule_prob

        root, unary, rule = roots(), terms(), rules()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()
            pass

        self.clear_metrics()  # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            # 'kl': torch.tensor(0, device=self.device)
        }

    def partition_function(self, max_length=200):
        return self.part(
            self.rules, lens=max_length, mode="depth", until_converge=True
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
        words = input["word"]
        # # Sequence permutate randomly
        # input["word"] = words[:, torch.randperm(words.shape[1])]

        # Calculate rule distributions
        self.rules = self.forward(input)
        # terms = torch.randint(0, self.V, input["word"].shape, device=self.device)
        # terms = self.term_from_unary(terms, self.rules["unary"])
        # terms = self.term_from_unary(
        #     input["word"], self.rules["unary"], smooth=self.smooth
        # )
        self.rules["word"] = input["word"]

        if partition:
            result = self.pcfg(
                self.rules, self.rules["unary"], lens=input["seq_len"], topk=1
            )
            # sent = self.pcfg(
            #     self.rules, terms, lens=input["seq_len"]
            # )
            self.pf = self.part(
                self.rules, lens=input["seq_len"], mode=self.mode
            )
            if soft:
                return (-result["partition"]), self.pf
                # return (-sent["partition"]), self.pf
            # result["partition"] = result["partition"] - sent["partition"]
            result["partition"] = result["partition"] - self.pf
        else:
            result = self.pcfg(
                self.rules,
                self.rules["unary"],
                lens=input["seq_len"],
                dropout=self.dropout,
            )

        return -result["partition"]

    def evaluate(self, input, decode_type, depth=0, label=False, **kwargs):
        self.rules = self.forward(input)

        b = input["word"].shape[0]
        lens = input["seq_len"]
        rules = {k: v.expand(b, *v.shape[1:]) for k, v in self.rules.items()}
        # terms = self.term_from_unary(input["word"], rules["unary"])

        if decode_type == "viterbi":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=lens,
                viterbi=True,
                mbr=False,
                dropout=self.dropout,
                label=label,
            )
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
