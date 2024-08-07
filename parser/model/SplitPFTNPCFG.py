from itertools import chain

import torch

from .PFTNPCFG import PFTNPCFG
from ..pcfgs.pcfg import PCFG


class SplitPFTNPCFG(PFTNPCFG):
    def __init__(self, args):
        args.V += 3
        super().__init__(args)

    def split_unknown(self, words, pos):
        n_words = words.clone()
        unk_idx = (words.flatten() == 1).nonzero().flatten().tolist()
        if len(unk_idx) != 0:
            pos_tags = list(chain.from_iterable(pos))
            new_tok = []
            for i in unk_idx:
                if "NN" in pos_tags[i]:
                    new_tok.append(10003)
                elif "JJ" in pos_tags[i]:
                    new_tok.append(10004)
                elif "VB" in pos_tags[i]:
                    new_tok.append(10005)
                else:
                    new_tok.append(1)
            unk_uidx = torch.unravel_index(
                torch.tensor(unk_idx).int(), words.shape
            )
            n_words[unk_uidx] = torch.tensor(new_tok, device=words.device)
        return n_words

    def loss(self, input, pos, **kwargs):
        words = input["word"]
        n_words = self.split_unknown(words, pos)

        b, seq_len = words.shape[:2]

        self.rules = self.forward()
        self.rules = self.batchify(self.rules, n_words)

        trees = None

        if self.pretrained_models:
            trees = []
            # Tree selected from pre-trained parser
            for parse in self.parse_trees:
                tree = words.new_tensor(
                    [parse.get(str(words[i].tolist())) for i in range(b)]
                )
                trees.append(tree)

            # Calculate span frequency in predicted trees
            trees = torch.stack(trees, dim=1)
            # # Concatenated mask
            trees = trees.reshape(b, -1, 2)

        if trees is not None:
            tree_mask = trees.new_zeros(b, seq_len + 1, seq_len + 1).float()
            for i, t in enumerate(trees):
                for s in t:
                    tree_mask[i, s[0], s[1]] += 1

            # Softmax mask
            if self.mask_mode == "soft":
                idx0, idx1 = torch.triu_indices(
                    seq_len + 1, seq_len + 1, offset=2
                ).unbind()
                masked_data = tree_mask[:, idx0, idx1]
                masked_data = masked_data.softmax(-1)
                tree_mask[:, idx0, idx1] = masked_data
            elif self.mask_mode == "hard":
                tree_mask = tree_mask > 0

        result = self.pcfg(
            self.rules,
            self.rules["unary"],
            lens=input["seq_len"],
            tree=tree_mask,
        )
        return -result["partition"].mean()

    def evaluate(
        self,
        input,
        pos,
        decode_type="mbr",
        depth=0,
        label=True,
        rule_update=False,
        **kwargs,
    ):
        if rule_update:
            need_update = True
        else:
            if hasattr(self, "rules"):
                need_update = False
            else:
                need_update = True

        if need_update:
            self.rules = self.forward()

        if decode_type == "viterbi":
            if not hasattr(self, "viterbi_pcfg"):
                self.viterbi_pcfg = PCFG()
                self.rules = self.compose(self.rules)

        n_words = self.split_unknown(input["word"], pos)
        rules = self.batchify(self.rules, n_words)

        if decode_type == "viterbi":
            result = self.viterbi_pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=True,
                mbr=False,
                label=label,
            )
        elif decode_type == "mbr":
            result = self.pcfg(
                rules,
                rules["unary"],
                lens=input["seq_len"],
                viterbi=False,
                mbr=True,
                label=label,
            )
        else:
            raise NotImplementedError

        result.update({"word": n_words})
        return result
