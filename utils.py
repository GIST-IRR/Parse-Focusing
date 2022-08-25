from nltk import Tree
import torch
import os
import matplotlib.pyplot as plt

def min_depth_for_len(length):
    return length.min().log2().ceil().long().item() + 1

def max_depth_for_len(length):
    return length.max().long().item()

def range_depth_for_len(length):
    return min_depth_for_len(length), max_depth_for_len(length)

def depth_to_onehot(length, depth):
    batch = length.shape[0]
    min_d, max_d = range_depth_for_len(length)
    size = max_d - min_d + 1
    idx = depth - min_d
    result = length.new_zeros((batch, size))
    result[torch.arange(batch), idx] = 1
    return result.float()

def depth_to_index(length, depth):
    min_d = min_depth_for_len(length)
    idx = depth - min_d
    return idx

def depth_from_span(span):
    tree = span_to_tree(span)
    return tree.height() + 1

def depth_from_tree(tree):
    return tree.height() + 1

def sort_span(span):
    span = [sorted(p, key=lambda x: x[1], reverse=True) for p in span]
    span = [sorted(p, key=lambda x: x[0]) for p in span]
    return span

def span_to_tree(span):
    return Tree.fromlist(span_to_list(span))

def span_to_list_old(span):
    root = span[0]
    label = f'NT-{root[2]}' if len(root) >= 3 else 'NT'
    if len(span) == 1:
        return [label, ['T']]
    left_child = span[1]
    if len(span) == 2:
        return [label, span_to_list([left_child])]
    others = span[2:]

    sibling_index = []
    end = left_child[1]
    for n in others:
        if n[0] >= end:
            idx = others.index(n)
            sibling_index.append(idx)
            end = n[1]

    children = []
    if len(sibling_index) > 0:
        left_child = [left_child] + others[:sibling_index[0]]
    else:
        left_child = [left_child] + others
    children.append(span_to_list(left_child))

    for i in range(len(sibling_index)):
        if i == len(sibling_index) - 1:
            child = others[sibling_index[i]:]
            children.append(span_to_list(child))
        else:
            child = others[sibling_index[i]:sibling_index[i+1]]
            children.append(span_to_list(child))
    return [label] + children

def span_to_list(span):
    root = span[0]
    
    if root[0] + 1 == root[1]:
        label = f'T-{root[2]}' if len(root) >= 3 else 'T'
        return [label, ['word']]

    label = f'NT-{root[2]}' if len(root) >= 3 else 'NT'
    left_child = span[1]
    if len(span) == 2:
        return [label, span_to_list([left_child])]
    others = span[2:]

    sibling_index = []
    end = left_child[1]
    for n in others:
        if n[0] >= end:
            idx = others.index(n)
            sibling_index.append(idx)
            end = n[1]

    children = []
    if len(sibling_index) > 0:
        left_child = [left_child] + others[:sibling_index[0]]
    else:
        left_child = [left_child] + others
    children.append(span_to_list(left_child))

    for i in range(len(sibling_index)):
        if i == len(sibling_index) - 1:
            child = others[sibling_index[i]:]
            children.append(span_to_list(child))
        else:
            child = others[sibling_index[i]:sibling_index[i+1]]
            children.append(span_to_list(child))
    return [label] + children

def tensor_to_heatmap(x, batch=True, dirname='heatmap', filename='cos_sim.png', vmin=-1, vmax=1):
    if batch:
        x = x.mean(0)
    x = x.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    pc = ax.pcolormesh(x, vmin=vmin, vmax=vmax, cmap='RdBu')
    fig.colorbar(pc, ax=ax)
    path = os.path.join(dirname, filename)
    plt.gca().invert_yaxis()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return fig

def save_rule_heatmap(rules, dirname='heatmap', filename='rules_prop.png', grad=False, root=True, rule=True, unary=True):
    if grad:
        root_data = rules['root'].grad[0].detach().cpu()
        rule_data = rules['rule'].grad[0].detach().cpu()
        unary_data = rules['unary'].grad[0].detach().cpu()
    else:
        root_data = rules['root'][0].detach().cpu()
        rule_data = rules['rule'][0].detach().cpu()
        unary_data = rules['unary'][0].detach().cpu()

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

    def gram_schmidt(vv):
            def projection(u, v):
                proj = (v * u).sum() / (u * u).sum() * u
                return proj.contiguous()

            n, d = vv.shape
            uu = vv.new_zeros(vv.shape)
            uu[0].copy_(vv[0])
            # uu[0].copy_(vv[0] / torch.linalg.norm(vv[0]))
            for k in range(1, n):
                # vk = vv[k].clone()
                uk = vv[k].clone()
                # uk = 0
                for j in range(0, k):
                    uk = uk - projection(uu[j].clone(), uk)
                uu[k].copy_(uk)
                # uu[k].copy_(uk / torch.linalg.norm(uk))
            # for k in range(nk):
            #     uk = uu[:, k].clone()
            #     uu[:, k] = uk / uk.norm()
            return uu.contiguous()