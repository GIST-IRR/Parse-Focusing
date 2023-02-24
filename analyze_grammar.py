import argparse
from pathlib import Path

import torch
from torch_support.train_support import (
    get_config_from,
)
from torch_support.load_model import get_model_args
from utils import save_rule_heatmap


def main(args):
    model_args = get_config_from(args.config)

    path = args.model_paths
    path = list(map(Path, path))
    # path = [p / 'rule_dist.pt' for p in path]

    # Single
    if len(path) == 1:
        model = get_model_args(model_args.model)
        with path[0].open('rb') as f:
            rules = torch.load(f, map_location='cpu')
        model.load_state_dict(rules['model'])
        model.eval()
        model.forward({"word": torch.tensor([[0]])})
        
        rules = {k: v.detach().cpu() for k, v in model.rules.items()}
        save_rule_heatmap(
            rules,
            filename=f'{path[0].parent.name}_rules.png'
        )

    # Comparing
    if len(path) == 2:
        rules = []
        for p in path:
            with p.open('rb') as f:
                rules.append(torch.load(f, map_location='cpu'))

        # ratio between two rule dist
        # negative: 0 is smaller than 1 ( 0 < 1): increase
        # positive: 1 is smaller than 0 ( 1 < 0): decrease
        # 0 - 1 : log scale : 0 / 1
        diff = {}
        for k in rules[0].keys():
            if k == 'kl':
                continue
            diff[k] = rules[0][k] - rules[1][k]
        
        save_rule_heatmap(diff, filename='german_bcl_rules.png')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/n_english_std_pretrained_nonterm.yaml')
    parser.add_argument('--model_paths', nargs='+', default='log/n_english_std_pretrained_nonterm/NPCFG2023-01-05-16_03_17/last.pt')
    args = parser.parse_args()

    # yaml_cfg = yaml.load(open(args.conf, 'r'))
    # args = edict(yaml_cfg)

    main(args)