import torch
import torch.nn as nn

from ..model.FNPCFG import FNPCFG


class RFNPCFG(FNPCFG):
    def __init__(self, args):
        super().__init__(args)

        roots = torch.tensor(torch.load(args.pretrained_root))
        unary = torch.tensor(torch.load(args.pretrained_unary))
        binary = torch.tensor(torch.load(args.pretrained_binary))

        roots = roots.log()
        unary = unary.log()
        binary = binary.log()

        self.roots = nn.Parameter(roots)
        self.unary = nn.Parameter(unary)
        self.binary = nn.Parameter(binary)

    def forward(self, input):
        return {
            "root": self.roots,
            "unary": self.unary,
            "rule": self.binary,
        }
