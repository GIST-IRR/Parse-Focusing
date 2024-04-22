import numpy as np
import torch
import torch.nn as nn

from ..pfs.partition_function import PartitionFunction
from ..pcfgs.pcfg import Faster_PCFG, PCFG
from ..model.NeuralPCFG import NeuralPCFG
from .PCFG_module import (
    Term_parameterizer,
    Nonterm_parameterizer,
    Root_parameterizer,
)


class FNPCFG(NeuralPCFG):
    """Embedding Related FGG-NPCFG"""

    def __init__(self, args):
        super(FNPCFG, self).__init__(args)
        self.pcfg = Faster_PCFG()

        self.activation = getattr(args, "activation", "relu")
        self.norm = getattr(args, "norm", None)

        self.terms = Term_parameterizer(
            self.s_dim,
            self.T,
            self.V,
            activation=self.activation,
            softmax=True,
            norm=self.norm,
        )
        self.nonterms = Nonterm_parameterizer(
            self.s_dim, self.NT, self.T, softmax=True, norm=self.norm
        )
        self.root = Root_parameterizer(
            self.s_dim, self.NT, activation=self.activation, norm=self.norm
        )

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()

    def forward(self):
        # Root
        root = self.root()
        # Rule
        rule = self.nonterms(reshape=True)
        # Unary
        unary = self.terms()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            rule.retain_grad()
            unary.retain_grad()

        self.clear_metrics()  # clear metrics becuase we have new rules

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            # 'kl': torch.tensor(0, device=self.device)
        }

    def batchify(self, rules, words):
        b = words.shape[0]

        root = rules["root"]
        root = root.expand(b, root.shape[-1])

        rule = rules["rule"]
        rule = rule.expand(b, *rule.shape)

        unary = rules["unary"]
        unary = unary[torch.arange(self.T)[None, None], words[:, :, None]]

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
        }

    def loss(
        self,
        input,
        partition=False,
        soft=False,
        parser=False,
        trees=True,
        *args,
        **kwargs,
    ):
        words = input["word"]

        # Calculate rule distributions
        self.rules = self.forward()
        self.rules = self.batchify(self.rules, words)

        result = self.pcfg(
            self.rules,
            self.rules["unary"],
            lens=input["seq_len"],
            dropout=self.dropout,
        )

        return -result["partition"]

    def evaluate(self, input, decode_type, depth=0, label=False, **kwargs):
        self.rules = self.forward()
        self.rules = self.batchify(self.rules, input["word"])

        b = input["word"].shape[0]
        lens = input["seq_len"]
        rules = {k: v.expand(b, *v.shape[1:]) for k, v in self.rules.items()}

        if decode_type == "viterbi":
            self.pcfg = PCFG()
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
