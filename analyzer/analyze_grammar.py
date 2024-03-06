import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_support.train_support import (
    get_config_from,
)
from torch_support.load_model import get_model_args
from torch_support.metric import *
from utils import save_rule_heatmap, tensor_to_heatmap


def draw_similarities(dist, emb, tag, dirname):
    # Diversity for rule distributions
    if dist is not None:
        nt = dist.shape[1]
        # KL Divergence
        kl_mat = torch.zeros(nt, nt)
        for i in range(nt):
            for j in range(nt):
                kl_mat[i, j] = kl_div(
                    dist[0, i].flatten(), dist[0, j].flatten()
                )
        tensor_to_heatmap(
            kl_mat,
            dirname=dirname,
            filename=f"heatmap_{tag}_kl.png",
            batch=False,
            vmin=0,
            vmax=kl_mat.max(),
        )

        # Normalized Mutual Information
        nmi_mat = torch.zeros(nt, nt)
        for i in range(nt):
            for j in range(nt):
                nmi_mat[i, j] = mutual_information(
                    dist[0, i].flatten().unsqueeze(0),
                    dist[0, j].flatten().unsqueeze(0),
                    normalize=True,
                )
        tensor_to_heatmap(
            nmi_mat,
            dirname=dirname,
            filename=f"heatmap_{tag}_mi.png",
            batch=False,
            vmin=0,
            vmax=1,
        )

    # Diversity for symbol embeddings
    if emb is not None:
        nt = emb.shape[0]
        # L2 distance
        l2_mat = torch.cdist(emb.unsqueeze(0), emb.unsqueeze(0)).squeeze(0)
        tensor_to_heatmap(
            l2_mat,
            dirname=dirname,
            filename=f"heatmap_{tag}_l2.png",
            batch=False,
            vmin=l2_mat.min(),
            vmax=l2_mat.max(),
        )

        # Cosine similarity
        cs_mat = pairwise_cosine_similarity(emb)
        tensor_to_heatmap(
            cs_mat,
            dirname=dirname,
            filename=f"heatmap_{tag}_cs.png",
            batch=False,
        )


def main(args):
    model_args = get_config_from(args.config)

    path = (
        args.model_paths
        if isinstance(args.model_paths, list)
        else [args.model_paths]
    )
    path = list(map(Path, path))
    # path = [p / 'rule_dist.pt' for p in path]

    # Single
    if len(path) == 1:
        model = get_model_args(model_args.model, device="cuda:0")
        with path[0].open("rb") as f:
            rules = torch.load(f, map_location="cuda:0")
        model.load_state_dict(rules["model"])
        model.eval()
        rules = model.forward({"word": torch.tensor([[0]])})

        # Diversity for rule distributions
        rule = rules["rule"].clone().detach()
        # rule_ent = rule[0].reshape(30, -1)
        # rule_ent = torch.sum(rule_ent.exp() * rule_ent, dim=1)

        # term = rules['unary'].clone().detach()
        # term_ent = term[0].reshape(60, -1)
        # term_ent = torch.sum(term_ent.exp() * term_ent, dim=1)

        try:
            rule_emb = model.nonterms.nonterm_emb.clone().detach()
        except:
            rule_emb = None
        draw_similarities(rule, rule_emb, "rule", f"{path[0].parent}")

        term = rules["unary"].clone().detach()
        try:
            term_emb = model.terms.term_emb.clone().detach()
        except:
            rule_emb = None
        draw_similarities(term, term_emb, "term", f"{path[0].parent}")
        # without embedding
        # draw_similarities(term, None, 'term', f'{path[0].parent}')

        print("done")

        # rules = {k: v.detach().cpu() for k, v in rules}
        # save_rule_heatmap(
        #     rules,
        #     filename=f'{path[0].parent.name}_rules.png'
        # )

    # Comparing
    if len(path) == 2:
        rules = []
        for p in path:
            with p.open("rb") as f:
                rules.append(torch.load(f, map_location="cpu"))

        # ratio between two rule dist
        # negative: 0 is smaller than 1 ( 0 < 1): increase
        # positive: 1 is smaller than 0 ( 1 < 0): decrease
        # 0 - 1 : log scale : 0 / 1
        diff = {}
        for k in rules[0].keys():
            if k == "kl":
                continue
            diff[k] = rules[0][k] - rules[1][k]

        save_rule_heatmap(diff, filename="german_bcl_rules.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/config.yaml",
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/last.pt",
    )
    args = parser.parse_args()

    # yaml_cfg = yaml.load(open(args.conf, 'r'))
    # args = edict(yaml_cfg)

    main(args)
