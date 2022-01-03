import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import argparse
import pickle

def get_rules(data):
    with open(data, 'rb') as f:
        data = pickle.load(f)
    
    root = torch.cat([r['root'].cpu() for r in data])
    rules = torch.cat([r['rule'].cpu() for r in data])
    return root, rules

def print_tsne(tsne, data, split, name):
    results = tsne.fit_transform(data.view(data.shape[0], -1).numpy())
    plt.figure()
    plt.scatter(results[:split, 0], results[:split, 1], c='blue', s=plt.rcParams['lines.markersize'])
    plt.scatter(results[split:split*2, 0], results[split:split*2, 1], c='red', s=plt.rcParams['lines.markersize'])
    plt.scatter(results[split*2:split*3, 0], results[split*2:split*3, 1], c='green', s=plt.rcParams['lines.markersize'])
    plt.scatter(results[split*3:, 0], results[split*3:, 1], c='yellow', s=plt.rcParams['lines.markersize'])
    plt.savefig(name)
    plt.cla()

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', required=True)
    # parser.add_argument('--other_data')
    args = parser.parse_args()

    o1_root, o1_rules = get_rules('orig_debug_0.pkl')
    o2_root, o2_rules = get_rules('orig_debug_1.pkl')
    # o3_root, o3_rules = get_rules('orig_debug_2.pkl')
    p_root, p_rules = get_rules('prop_debug_0.pkl')
    p1_root, p1_rules = get_rules('prop_debug_1.pkl')

    split = p_rules.shape[0]
    root = torch.cat([o1_root, o2_root, p_root, p1_root])
    rules = torch.cat([o1_rules, o2_rules, p_rules, p1_rules])

    tsne = TSNE(n_components=2)

    print_tsne(tsne, root, split, 'root_oopp.png')
    print_tsne(tsne, rules, split, 'rule_oopp.png')

    plt.close()

if __name__=='__main__':
    main()