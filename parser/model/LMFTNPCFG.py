from collections import defaultdict

import torch

from parser.model.FNPCFG import FNPCFG


class LMFTNPCFG(FNPCFG):
    """Labeled Parse-Focused TN-PCFG"""

    def __init__(self, args):
        super(LMFTNPCFG, self).__init__(args)
        self.mask_mode = getattr(args, "mask_mode", "soft")
        self.pretrained_models = getattr(args, "pretrained_models", None)
        self.parse_trees = []
        self.parse_labels = []
        self.label_set = set()
        if self.pretrained_models:
            for model in self.pretrained_models:
                parses = defaultdict(list)
                labels = defaultdict(list)
                with open(model, "rb") as f:
                    pts = torch.load(f)
                for t in pts:
                    n_word = t["word"]
                    label = t["label"]
                    if "pred_tree" in t.keys():
                        n_tree = [s for s in t["pred_tree"] if s[1] - s[0] > 1]
                    elif "tree" in t.keys():
                        n_tree = [s[:2] for s in t["tree"] if s[1] - s[0] > 1]
                    parses[str(n_word)] = n_tree
                    labels[str(n_word)] = label
                    self.label_set.update(label)
                self.parse_trees.append(parses)
                self.parse_labels.append(labels)

            self.label_set = list(sorted(self.label_set))
            self.label2idx = {l: i for i, l in enumerate(self.label_set)}

    def loss(self, input, partition=False, soft=False, label=False, **kwargs):
        words = input["word"]
        b, seq_len = words.shape[:2]

        self.rules = self.forward()
        self.rules = self.batchify(self.rules, words)

        trees = None

        if self.pretrained_models:
            trees = []
            # Tree selected from pre-trained parser
            for parse in self.parse_trees:
                tree = words.new_tensor(
                    [parse.get(str(words[i].tolist())) for i in range(b)]
                )
                trees.append(tree)

            labels = []
            for parse in self.parse_labels:
                ls = [parse.get(str(words[i].tolist())) for i in range(b)]
                ls = words.new_tensor(
                    [[self.label2idx.get(e) for e in l] for l in ls]
                )
                labels.append(ls)

            # Calculate span frequency in predicted trees
            trees = torch.stack(trees, dim=1)
            labels = torch.stack(labels, dim=1)
            trees = torch.cat([trees, labels.unsqueeze(-1)], dim=-1)
            # # Concatenated mask
            trees = trees.reshape(b, -1, 3)

        if trees is not None:
            tree_mask = trees.new_zeros(
                b, seq_len + 1, seq_len + 1, self.NT
            ).float()
            for i, t in enumerate(trees):
                for s in t:
                    tree_mask[i, s[0], s[1], s[2]] += 1

            # Softmax mask
            if self.mask_mode == "soft":
                idx0, idx1 = torch.triu_indices(
                    seq_len + 1, seq_len + 1, offset=2
                ).unbind()
                masked_data = tree_mask[:, idx0, idx1]
                masked_data = masked_data.reshape(b, -1)
                # Normalization
                masked_data = masked_data.softmax(-1)
                # masked_data = masked_data / masked_data.sum(-1, keepdim=True)
                # End
                masked_data = masked_data.reshape(b, -1, self.NT)
                tree_mask[:, idx0, idx1] = masked_data
            elif self.mask_mode == "hard":
                tree_mask = tree_mask > 0
            elif self.mask_mode == "no_process":
                pass

        result = self.pcfg(
            self.rules,
            self.rules["unary"],
            lens=input["seq_len"],
            tree=tree_mask,
        )
        return -result["partition"].mean()
