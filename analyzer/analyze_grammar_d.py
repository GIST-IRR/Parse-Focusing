import argparse
from pathlib import Path

import torch

from torch_support.metric import *
from utils import save_rule_heatmap, save_rule_ent_heatmap


def load_grammar(dir, tag):
    if not isinstance(dir, Path):
        dir = Path(dir)

    root_path = dir / f'{tag}_root_pd.pt'
    rule_path = dir / f'{tag}_rule_pd.pt'
    term_path = dir / f'{tag}_term_pd.pt'
    
    with root_path.open('rb') as f:
        root = torch.load(f)
    with rule_path.open('rb') as f:
        rule = torch.load(f)
    with term_path.open('rb') as f:
        term = torch.load(f)

    return root, rule, term

def norm(x, dim=None, eps=-15, log=False):
        if dim is None:
            x = (x / x.sum())
        else:
            x = (x / x.sum(dim, keepdims=True))
        if log:
            x = x.log()
            x = x.where(
                ~torch.logical_or(x.isinf(), x.isnan()),
                torch.full_like(x, eps)
            )
        else:
            x = x.where(
                ~torch.logical_or(x.isinf(), x.isnan()),
                torch.full_like(x, 0.)
            )
        return x

def main(args):
    root, rule, unary = load_grammar(args.dir, args.tag)
    rule_shape = rule.shape
    
    rule = rule.reshape(rule_shape[0], -1)

    log = True
    
    joint_rules = {
        "root": norm(root, log=log),
        "rule": norm(rule, log=log).reshape(1, *rule_shape),
        "unary": norm(unary, log=log).unsqueeze(0),
    }

    lhs2rhs = {
        "root": norm(root, dim=1, log=log),
        "rule": norm(rule, dim=1, log=log).reshape(1, *rule_shape),
        "unary": norm(unary, dim=1, log=log).unsqueeze(0),
    }

    rhs2lhs = {
        "root": norm(root, dim=0, log=log),
        "rule": norm(rule, dim=0, log=log).reshape(1, *rule_shape),
        "unary": norm(unary, dim=0, log=log).unsqueeze(0),
    }

    save_rule_heatmap(joint_rules, filename=f'{args.tag}_joint.png')
    save_rule_heatmap(lhs2rhs, filename=f'{args.tag}_lhs2rhs.png')
    save_rule_heatmap(rhs2lhs, filename=f'{args.tag}_rhs2lhs.png')
    
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='weights')
    parser.add_argument('--tag', default='basque_xbar_left')
    args = parser.parse_args()

    main(args)