import argparse
from pathlib import Path

import torch
from torch_support.train_support import (
    get_config_from,
)
from torch_support.load_model import get_model_args
import torch_support.load_model as load_model
from torch_support.metric import *
from utils import tensor_to_heatmap


def draw_similarities(emb, type="KL", tag="rule", dirname=""):
    # Diversity for rule distributions
    if type == "KL":
        # KL Divergence
        nt = emb.shape[1]
        mat = torch.zeros(nt, nt)
        for i in range(nt):
            for j in range(nt):
                mat[i, j] = kl_div(emb[0, i].flatten(), emb[0, j].flatten())
        vmin = 0
        vmax = mat.max()
    elif type == "MI":
        # Normalized Mutual Information
        nt = emb.shape[1]
        mat = torch.zeros(nt, nt)
        for i in range(nt):
            for j in range(nt):
                mat[i, j] = mutual_information(
                    emb[0, i].flatten().unsqueeze(0),
                    emb[0, j].flatten().unsqueeze(0),
                    normalize=True,
                )
        vmin = 0
        vmax = 1
    # Diversity for symbol embeddings
    elif type == "L2":
        # L2 distance
        mat = torch.cdist(emb.unsqueeze(0), emb.unsqueeze(0)).squeeze(0)
        vmin = mat.min()
        vmax = mat.max()
    elif type == "CS":
        # Cosine similarity
        mat = pairwise_cosine_similarity(emb)
        vmin = None
        vmax = None
    tensor_to_heatmap(
        mat,
        dirname=dirname,
        filename=f"heatmap_{tag}_{type.lower()}.png",
        batch=False,
        vmin=vmin,
        vmax=vmax,
    )


def main(config, model_path):
    model_args = get_config_from(config)

    # Get rule distribution from model
    # Model에서 rule distribution을 추출하는 코드를 분리하는게 좋지 않나?
    load_model.set_model_dir("parser/model")
    model = get_model_args(model_args.model, device="cuda:0")
    with model_path.open("rb") as f:
        rules = torch.load(f, map_location="cuda:0")
    model.load_state_dict(rules["model"])
    model.eval()
    # rules = model.forward({"word": torch.tensor([[0]])})
    rules = model.forward()

    # Diversity for rule distributions
    if "TNPCFG" in model_args.model.name:
        rules = model.compose(rules)
    rule = rules["rule"].clone().detach()
    rules["rule"] = rules["rule"].clamp(min=-35)

    # Calculate entropy
    # rule_ent = rule[0].reshape(30, -1)
    # rule_ent = torch.sum(rule_ent.exp() * rule_ent, dim=1)

    # term = rules['unary'].clone().detach()
    # term_ent = term[0].reshape(60, -1)
    # term_ent = torch.sum(term_ent.exp() * term_ent, dim=1)

    try:
        rule_emb = model.nonterms.nonterm_emb.clone().detach()
    except:
        rule_emb = None
    draw_similarities(rule, "KL", "rule", f"{model_path.parent}")
    draw_similarities(rule, "MI", "rule", f"{model_path.parent}")
    draw_similarities(rule_emb, "L2", "rule", f"{model_path.parent}")
    draw_similarities(rule_emb, "CS", "rule", f"{model_path.parent}")

    term = rules["unary"].clone().detach()
    try:
        term_emb = model.terms.term_emb.clone().detach()
    except:
        rule_emb = None
    draw_similarities(term, "KL", "term", f"{model_path.parent}")
    draw_similarities(term, "MI", "term", f"{model_path.parent}")
    draw_similarities(term_emb, "L2", "term", f"{model_path.parent}")
    draw_similarities(term_emb, "CS", "term", f"{model_path.parent}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/config.yaml",
    )
    parser.add_argument(
        "--model_path",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/last.pt",
        type=Path,
    )
    args = parser.parse_args()

    main(args.config, args.model_path)
