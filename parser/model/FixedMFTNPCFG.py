from pathlib import Path
from parser.model.PFTNPCFG import PFTNPCFG
import torch
import pickle
from torch import nn


class FixedMFTNPCFG(PFTNPCFG):
    def __init__(self, args):
        self.pretrained_term = getattr(args, "pretrained_term", None)
        with Path(self.pretrained_term).open("rb") as f:
            pretrained_term = pickle.load(f)
        pretrained_term = torch.tensor(pretrained_term)
        pretrained_term = pretrained_term.float().log().clamp(-1e9)

        args.T = pretrained_term.shape[0]
        super(FixedMFTNPCFG, self).__init__(args)

        self.register_buffer("term_prob", pretrained_term)

    def forward(self, **kwargs):
        root = self.root()
        unary = self.term_prob
        head, left, right = self.nonterms()

        return {
            "unary": unary,
            "root": root,
            "head": head,
            "left": left,
            "right": right,
        }
