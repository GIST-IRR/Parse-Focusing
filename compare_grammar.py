import argparse
from pathlib import Path
import os

import torch
import matplotlib.pyplot as plt


def save_rule_heatmap(rules, dirname='heatmap', filename='rules_prop.png', root=True, rule=True, unary=True):
    root_data = rules['root'][0]
    rule_data = rules['rule'][0]
    unary_data = rules['unary'][0]

    # plt.rcParams['figure.figsize'] = (70, 50)
    root_dfs = root_data.unsqueeze(0).numpy()
    rule_dfs = [r.numpy() for r in rule_data]
    unary_dfs = unary_data.numpy()
    # min max in seed
    if root:
        vmin = root_data.min()
        vmax = root_data.max()
        fig, ax = plt.subplots(figsize=(10, 5))
        pc = ax.pcolormesh(root_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'root_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # min max in local
    if rule:
        vmin = rule_data.min()
        vmax = rule_data.max()
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(70, 50))
        for df, ax in zip(rule_dfs, axes.flat):
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'rule_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # absolute min max
    if unary:
        vmin = unary_data.min()
        vmax = unary_data.max()
        fig, ax = plt.subplots(figsize=(20, 5))
        pc = ax.pcolormesh(unary_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f'unary_{filename}')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

def main(args):
    path = [args.model_path1, args.model_path2]
    path = map(Path, path)
    path = [p / 'rule_dist.pt' for p in path]
    rules = []
    for p in path:
        with p.open('rb') as f:
            rules.append(torch.load(f, map_location='cpu'))

    diff = {}
    for k in rules[0].keys():
        if k == 'kl':
            continue
        diff[k] = rules[0][k] - rules[1][k]
    # 0 - 1 : log scale : 0 / 1
    # ratio between two rule dist
    # negative: 0 is smaller than 1 ( 0 < 1): increase
    # positive: 1 is smaller than 0 ( 1 < 0): decrease
    
    save_rule_heatmap(diff, filename='german_bcl_rules.png')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path1', default='')
    parser.add_argument('--model_path2', default='')
    args = parser.parse_args()

    # yaml_cfg = yaml.load(open(args.conf, 'r'))
    # args = edict(yaml_cfg)

    main(args)