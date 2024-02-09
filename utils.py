from typing import Any
from nltk import Tree
import numpy as np
import torch
import os
from pathlib import Path

from collections import Counter, defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from sklearn.manifold import TSNE
import networkx as nx


def outlier(data, method="quantile", threshold=3.0):
    n_p, n_c = data.shape[:2]
    data = np.exp(data).reshape(n_p, -1)
    if method == "z_score":
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        z_score = (data - mean) / std
        out_idx = np.where(np.abs(z_score) > threshold)
        out = data[out_idx]

        c_idx = np.stack(np.unravel_index(out_idx[1], (n_c, n_c)), axis=-1)
        out_idx = np.concatenate([out_idx[0][:, None], c_idx], axis=-1)

        return [i for i, z in enumerate(z_score) if abs(z) > threshold]
    elif method == "quantile":
        q1 = np.percentile(data, 25, axis=-1, keepdims=True)
        q3 = np.percentile(data, 75, axis=-1, keepdims=True)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return [
            i for i, d in enumerate(data) if d < lower_bound or d > upper_bound
        ]
    else:
        raise ValueError("method must be z_score or quantile")


def clean_word(words):
    import re

    def clean_number(w):
        new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
        return new_w

    return [clean_number(word.lower()) for word in words]


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
    label = f"NT-{root[2]}" if len(root) >= 3 else "NT"
    if len(span) == 1:
        return [label, ["T"]]
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
        left_child = [left_child] + others[: sibling_index[0]]
    else:
        left_child = [left_child] + others
    children.append(span_to_list(left_child))

    for i in range(len(sibling_index)):
        if i == len(sibling_index) - 1:
            child = others[sibling_index[i] :]
            children.append(span_to_list(child))
        else:
            child = others[sibling_index[i] : sibling_index[i + 1]]
            children.append(span_to_list(child))
    return [label] + children


def span_to_list(span):
    root = span[0]
    start, end = root[:2]

    # Check trivial span
    if root[0] + 1 == root[1]:
        label = f"T-{root[2]}" if len(root) >= 3 else "T"
        return [label, ["word"]]

    label = f"NT-{root[2]}" if len(root) >= 3 else "NT"
    # Check single span
    if len(span) == 1:
        size = end - start
        children = [["T", ["word"]] for _ in range(size)]
        return [label, *children]

    # Check set of span for each children
    others = span[1:]
    children_index = []
    child_end = others[0][1]
    for n in others:
        if child_end == end:
            break
        if n[0] >= child_end:
            idx = others.index(n)
            children_index.append(idx)
            child_end = n[1]
    children_index = [0] + children_index + [len(others)]

    # Split by children index
    children = [
        others[children_index[i] : children_index[i + 1]]
        for i in range(len(children_index) - 1)
    ]
    children = [[[start, start]]] + children + [[[end, end]]]

    # If unseen terminal in tree, add them to children
    terms = {}
    for i in range(len(children) - 1):
        prev_end = children[i][0][1]
        size = children[i + 1][0][0] - prev_end
        if size > 0:
            term = [[i, i + 1] for i in range(prev_end, prev_end + size)]
            terms.update({i: term})
    children = children[1:-1]

    n_children = []
    for i in range(len(children)):
        if i in terms.keys():
            for t in terms[i]:
                n_children.append([t])
        n_children += [children[i]]
    if i + 1 in terms.keys():
        for t in terms[i + 1]:
            n_children.append([t])
    children = n_children

    children = [span_to_list(c) for c in children]

    return [label] + children


def tree_to_span(tree):
    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i + 1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, "NULL"]] + spans
        return j, spans

    return track(tree, 0)[1]


def tree_branching(length):
    return


def generate_random_span_by_length(
    length, gap=0, left_idx=None, right_idx=None
):
    if isinstance(length, int):
        dist = torch.randn(length)
        dist[-1] = 1e9
    else:
        dist = length
    if left_idx is None:
        left_idx = 0
    if right_idx is None:
        right_idx = len(dist)

    span = [(left_idx, right_idx)]

    if len(dist) != 1:
        max_idx = dist[:-1].argmax().item()
        assert dist[-1] > dist[max_idx]
        left = dist[: max_idx + 1]
        right = dist[max_idx + 1 :]
        span += generate_random_span_by_length(
            left, gap, left_idx, left_idx + max_idx + 1
        )
        span += generate_random_span_by_length(
            right, gap, left_idx + max_idx + 1, right_idx
        )

    return span


def tensor_to_heatmap(
    x, batch=True, dirname="heatmap", filename="cos_sim.png", vmin=-1, vmax=1
):
    if batch:
        x = x.mean(0)
    x = x.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    pc = ax.pcolormesh(x, vmin=vmin, vmax=vmax, cmap="RdBu")
    fig.colorbar(pc, ax=ax)
    path = os.path.join(dirname, filename)
    plt.gca().invert_yaxis()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return fig


def save_rule_heatmap(
    rules,
    dirname="heatmap",
    filename="rules_prop.png",
    grad=False,
    root=True,
    rule=True,
    unary=True,
    abs=False,
    local=False,
    batched=True,
):
    if grad:
        root_data = rules["root"].grad.detach().cpu()
        rule_data = rules["rule"].grad.detach().cpu()
        unary_data = rules["unary"].grad.detach().cpu()
    else:
        root_data = rules["root"].detach().cpu()
        rule_data = rules["rule"].detach().cpu()
        unary_data = rules["unary"].detach().cpu()

    root_data = root_data[0]
    if batched:
        rule_data = rule_data[0]
        unary_data = unary_data[0]

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
        path = os.path.join(dirname, f"root_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # min max in local
    if rule:
        # absolute min max
        if abs:
            vmin = -100.0
            vmax = 0.0
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"global_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # min max in local
        if local:
            vmin = rules.min()
            vmax = rules.max()
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"local_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(70, 50))
        for df, ax in zip(rule_dfs, axes.flat):
            vmin = df.min()
            vmax = df.max()
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"rule_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # absolute min max
    if unary:
        vmin = unary_data.min()
        vmax = unary_data.max()
        fig, ax = plt.subplots(figsize=(30, 5))
        pc = ax.pcolormesh(unary_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"unary_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def save_rule_heatmap_raw(
    model,
    dirname="heatmap",
    filename="rules_prop.png",
    root=True,
    rule=True,
    unary=True,
    abs=False,
    local=False,
):
    # min max in seed
    if root:
        root_data = model.root().detach().cpu()
        root_dfs = root_data.numpy()

        vmin = root_data.min()
        vmax = root_data.max()
        fig, ax = plt.subplots(figsize=(10, 5))
        pc = ax.pcolormesh(root_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"root_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # min max in local
    if rule:
        rule_data = model.nonterms(reshape=True)
        if not isinstance(rule_data, tuple):
            rule_data = rule_data.detach().cpu()
            rule_dfs = rule_data.numpy()
        else:
            rule = False
        # absolute min max
        if abs:
            vmin = -10.0
            vmax = 0.0
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"global_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # min max in local
        if local:
            vmin = rule_data.min()
            vmax = rule_data.max()
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"local_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(70, 50))
        for df, ax in zip(rule_dfs, axes.flat):
            # vmin = df.min()
            # vmax = df.max()
            vmin = -20.0
            vmax = 0
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"rule_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # absolute min max
    if unary:
        unary_data = model.terms().detach().cpu()
        unary_dfs = unary_data.numpy()
        # vmin = unary_data.min()
        # vmax = unary_data.max()
        vmin = -20.0
        vmax = 0
        fig, ax = plt.subplots(figsize=(50, 5))
        pc = ax.pcolormesh(unary_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"unary_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def save_rule_distribution_raw(
    model,
    dirname="heatmap",
    filename="rules_prop.png",
    root=True,
    rule=True,
    unary=True,
    abs=False,
    local=False,
):
    # min max in seed
    if root:
        root_data = model.root().detach().cpu()
        root_dfs = root_data.numpy()
        raise NotImplementedError

    # min max in local
    if rule:
        try:
            rule_data = model.nonterms(reshape=True)
        except:
            rule_data = model.nonterms()

        if not isinstance(rule_data, tuple):
            rule_data = rule_data.detach().cpu()
            rule_dfs = rule_data.numpy()
        else:
            rule = False
        rule_dfs = np.exp(rule_dfs)

        fig, axs = plt.subplots(5, 6, figsize=(70, 50))
        for i, ax in enumerate(axs.flatten()):
            dfs = rule_dfs[i].flatten()
            ax.plot(np.arange(0, dfs.shape[0]), dfs)
            ax.set_xlabel("rule index")
            ax.set_ylabel("probability")
            ax.grid(True)

            max_prob = dfs.max()
            max_idx = dfs.argmax()
            ax.annotate(
                f"{max_idx}, {max_prob}",
                xy=(max_idx, max_prob),
                xytext=(5, 0),
                textcoords="offset points",
                horizontalalignment="left",
                verticalalignment="bottom" if max_prob > 0 else "top",
            )
        path = os.path.join(dirname, f"rule_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # absolute min max
    if unary:
        unary_data = model.terms().detach().cpu()
        unary_dfs = unary_data.numpy()
        unary_dfs = np.exp(unary_dfs)
        # vmin = unary_data.min()
        # vmax = unary_data.max()
        vmin = 0.0
        vmax = 1.0
        fig, axs = plt.subplots(6, 10, figsize=(55, 30))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(np.arange(0, unary_dfs.shape[1]), unary_dfs[i])
            ax.set_xlabel("word index")
            ax.set_ylabel("probability")
            ax.grid(True)

            max_prob = unary_dfs[i].max()
            max_idx = unary_dfs[i].argmax()
            ax.annotate(
                f"{max_idx}, {max_prob}",
                xy=(max_idx, max_prob),
                xytext=(5, 0),
                textcoords="offset points",
                horizontalalignment="left",
                verticalalignment="bottom" if max_prob > 0 else "top",
            )
        path = os.path.join(dirname, f"unary_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def save_rule_ent_heatmap(
    rules,
    dirname="heatmap",
    filename="rules_prop.png",
    rule=True,
    unary=True,
    abs=False,
    local=False,
):
    rule_data = rules["C2N"].detach().cpu()
    unary_data = rules["w2T"].detach().cpu()

    # plt.rcParams['figure.figsize'] = (70, 50)
    rule_dfs = [r.numpy() for r in rule_data]
    unary_dfs = unary_data.numpy()

    # min max in local
    if rule:
        # absolute min max
        if abs:
            vmin = -100.0
            vmax = 0.0
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"global_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # min max in local
        if local:
            vmin = rules.min()
            vmax = rules.max()
            fig, axes = plt.subplots(nrows=5, ncols=6)
            for df, ax in zip(rule_dfs, axes.flat):
                pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax)
            path = os.path.join(dirname, f"local_{filename}")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(70, 50))
        for df, ax in zip(rule_dfs, axes.flat):
            vmin = df.min()
            vmax = df.max()
            pc = ax.pcolormesh(df, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"c2n_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # absolute min max
    if unary:
        vmin = unary_data.min()
        vmax = unary_data.max()
        fig, ax = plt.subplots(figsize=(30, 5))
        pc = ax.pcolormesh(unary_dfs, vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax)
        path = os.path.join(dirname, f"w2t_{filename}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def count_recursive_rules(trees):
    assert isinstance(trees, list) and isinstance(
        trees[0], Tree
    ), "trees must be a list of nltk.Tree"

    rule_counter = Counter()
    nonterm_counter = Counter()
    rec_counter = Counter()

    def contain_child_nonterminal(production):
        for child in production.rhs():
            if "NT" in child.symbol().split("-")[0]:
                return True
        return False

    def is_recursive(production):
        return any([production.lhs() == c for c in production.rhs()])

    for tree in trees:
        binary_prod = [p for p in tree.productions() if len(p.rhs()) == 2]
        nonterm_prod = [p for p in binary_prod if contain_child_nonterminal(p)]
        rec_prod = [p for p in nonterm_prod if is_recursive(p)]
        rule_counter.update(binary_prod)
        nonterm_counter.update(nonterm_prod)
        rec_counter.update(rec_prod)

    return rule_counter, nonterm_counter, rec_counter


def save_correspondence(
    d,
    threshold=7,
    threshold_label=30,
    dirname="heatmap",
    filename="correspondence.png",
):
    if len(d.items()) <= 0:
        return

    n_labels = [0 for _ in range(len(list(d.values())[0]))]
    for k, v in d.items():
        for i in range(len(v)):
            n_labels[i] += v[i]
    n_labels = [n / sum(n_labels) for n in n_labels]
    label_str = ""
    for i, n in enumerate(n_labels):
        label_str += f"{i}: {n:.4f} "
    entropy = -sum([n * np.log(n) for n in n_labels if n != 0])
    label_str += f"Entropy: {entropy}"
    print(label_str.strip())

    if len(n_labels) > 30:
        return

    threshold = threshold if len(d) > threshold else len(d)
    d = sorted(d.items(), key=lambda x: sum(x[1]), reverse=True)[:threshold]

    x_ticks_label = list(range(len(d[0][1])))
    x_ticks = [i + 0.5 for i in x_ticks_label]
    y_ticks_label = [e[0] for e in d]
    y_ticks = [i + 0.5 for i in list(range(threshold))]

    d = [[n / sum(e[1]) for n in e[1]] for e in d]

    row_size = int(len(x_ticks) / 2) + 2
    col_size = int(threshold / 2)
    fig, ax = plt.subplots(figsize=(row_size, col_size))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks_label)
    ax.set_yticklabels(y_ticks_label)

    pc = ax.pcolormesh(d, vmin=0, cmap="YlGnBu")
    fig.colorbar(pc, ax=ax)
    path = Path(dirname, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def rule_frequency(model, dirname, filename="rule_frequency.png"):
    pass


def visualize_embeddings(
    model, dirname, filename="embeddings.png", label=True
):
    try:
        root_emb = model.root_emb
        nonterm_emb = model.nonterm_emb
        term_emb = model.term_emb
    except:
        root_emb = model.root.root_emb
        nonterm_emb = model.nonterms.nonterm_emb
        term_emb = model.terms.term_emb

    root_label = ["ROOT"]
    NT_label = ["NT-" + str(i) for i in range(len(nonterm_emb))]
    T_label = ["T-" + str(i) for i in range(len(term_emb))]
    g_label = root_label + NT_label + T_label

    data = (
        torch.cat([root_emb, nonterm_emb, term_emb], dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
    )
    reduced_data = tsne.fit_transform(data)

    root_data = reduced_data[0]
    nonterm_data = reduced_data[1 : len(NT_label) + 1]
    term_data = reduced_data[len(NT_label) + 1 :]

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
    ax.scatter(root_data[0], root_data[1])
    ax.scatter(nonterm_data[:, 0], nonterm_data[:, 1])
    ax.scatter(term_data[:, 0], term_data[:, 1])

    if label:
        for i, txt in enumerate(g_label):
            ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))
    path = Path(dirname, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def visualize_probability_distribution(
    rule, dirname, filename="rule_emb.png", label=True
):
    nt, _ = rule.shape
    # Filtering rules that bigger than mean
    rule = rule[:, torch.where(rule > rule.mean(), 1, 0).any(0)]
    label = ["NT-" + str(i) for i in range(nt)]

    data = rule.detach().cpu().numpy()

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=2000)
    reduced_data = tsne.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1])

    if label:
        for i, txt in enumerate(label):
            ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))

    path = Path(dirname, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def visualize_graph(trees, dirname, filename="graph.png"):
    prods = []
    for tree in trees:
        prods += tree.productions()
    prod_count = Counter(prods)

    nodes = set()
    edges = defaultdict(int)
    for k, v in prod_count.items():
        parent, children = str(k).split(" -> ")

        parent = parent[1:-1]
        if parent.split("-")[0] == "T":
            continue
        nodes.add(parent)

        children = children.split(" ")
        children = [c[1:-1] for c in children]
        nodes.update(children)

        if len(children) > 1:
            lk = f"{parent} -> {children[0]}"
            rk = f"{parent} -> {children[1]}"
            edges[(parent, children[0])] += v
            edges[(parent, children[1])] += v
        else:
            nk = f"{parent} -> {children[0]}"
            edges[(parent, children[0])] += v

    nonterminal_nodes = [n for n in nodes if n.split("-")[0] == "NT"]
    terminal_nodes = [n for n in nodes if n.split("-")[0] == "T"]
    edges = [(k[0], k[1], v) for k, v in edges.items()]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    shells = [nonterminal_nodes, terminal_nodes]

    w_total = sum(weights)
    weights = [w / w_total for w in weights]

    fig, ax = plt.subplots(figsize=(20, 20))
    # Draw Nonterminal Nodes
    nx.draw_networkx_nodes(
        G,
        # pos=nx.shell_layout(G),
        # pos=nx.kamada_kawai_layout(G),
        # pos=nx.spectral_layout(G),
        pos=nx.shell_layout(G, shells),
        nodelist=nonterminal_nodes,
        node_color="#FF1714",
        node_size=500,
    )
    # Draw Terminal Nodes
    nx.draw_networkx_nodes(
        G,
        # pos=nx.kamada_kawai_layout(G),
        # pos=nx.spectral_layout(G),
        pos=nx.shell_layout(G, shells),
        nodelist=terminal_nodes,
        node_color="#673A85",
        node_size=500,
    )
    # Draw edges
    nx.draw_networkx_labels(
        G,
        # pos=nx.shell_layout(G),
        # pos=nx.kamada_kawai_layout(G),
        # pos=nx.spectral_layout(G),
        pos=nx.shell_layout(G, shells),
        font_size=10,
    )
    alpha = [w / max(weights) for w in weights]
    nx.draw_networkx_edges(
        G,
        # pos=nx.shell_layout(G),
        # pos=nx.kamada_kawai_layout(G),
        # pos=nx.spectral_layout(G),
        pos=nx.shell_layout(G, shells),
        edgelist=edges,
        edge_color=weights,
        alpha=alpha,
        width=2,
        edge_cmap=mpl.colormaps["viridis"],
    )
    path = Path(dirname, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def visualize_rule_graph(trees, dirname, filename="rule_graph.png"):
    prods = []
    for tree in trees:
        prods += tree.productions()
    prod_count = Counter(prods)

    parent_nodes = set()
    children_nodes = set()
    edges = set()
    for k, v in prod_count.items():
        parent, children = str(k).split(" -> ")

        parent = parent[1:-1]
        if parent.split("-")[0] == "T":
            continue
        parent_nodes.add(parent)

        children = children.replace("'", "")
        children_nodes.add(children)

        edges.add((parent, children, v))

    nodes = parent_nodes.union(children_nodes)
    shells = [parent_nodes, children_nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())

    w_total = sum(weights)
    weights = [w / w_total for w in weights]

    fig, ax = plt.subplots(figsize=(20, 20))
    # Draw Nonterminal Nodes
    nx.draw_networkx_nodes(
        G,
        pos=nx.shell_layout(G, shells),
        nodelist=parent_nodes,
        node_color="#FF1714",
        node_size=500,
    )
    # Draw Terminal Nodes
    nx.draw_networkx_nodes(
        G,
        pos=nx.shell_layout(G, shells),
        nodelist=children_nodes,
        node_color="#673A85",
        node_size=500,
    )
    # Draw edges
    nx.draw_networkx_labels(
        G,
        pos=nx.shell_layout(G, shells),
        font_size=10,
    )
    alpha = [w / max(weights) for w in weights]
    nx.draw_networkx_edges(
        G,
        pos=nx.shell_layout(G, shells),
        edgelist=edges,
        edge_color=weights,
        alpha=alpha,
        width=2,
        edge_cmap=mpl.colormaps["viridis"],
    )
    path = Path(dirname, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def visualize_dist_change(dist_list):
    def polygon_under_graph(x, y):
        return [(x[0], 0), *zip(x, y), (x[-1], 0)]

    step, batch, size = dist_list.shape
    dist_list = np.transpose(dist_list, (1, 0, 2))
    fig, axes = plt.figure(nrows=5, ncols=6, figsize=(70, 50))

    for dist, ax in zip(dist_list, axes.flat):
        x = np.arange(batch)
        verts = [
            polygon_under_graph(
                x,
            )
            for ds in dist_list
        ]
        facecolors = plt.colormaps["viridis_r"](np.linspace(0, 1, len(verts)))
        poly = PolyCollection(verts, facecolors=facecolors)
        ax.add_collection3d(poly, zs=step, zdir="y")
    # TODO: WORKING


def gram_schmidt(vv):
    def projection(u, v):
        proj = (v * u).sum() / (u * u).sum() * u
        return proj.contiguous()

    n, d = vv.shape
    uu = vv.new_zeros(vv.shape)
    # uu[0].copy_(vv[0])
    uu[0].copy_(vv[0] / torch.linalg.norm(vv[0]))
    for k in range(1, n):
        # vk = vv[k].clone()
        uk = vv[k].clone()
        # uk = 0
        for j in range(0, k):
            uk = uk - projection(uu[j].clone(), uk)
        # uu[k].copy_(uk)
        uu[k].copy_(uk / torch.linalg.norm(uk))
    for k in range(d):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu.contiguous()


class FrequencyCache:
    def __init__(self, func) -> None:
        self.func = func
        self.dict = {}

    def __call__(self, *args, **kwds) -> Any:
        if args in self.dict:
            return self.dict[args]
        else:
            res = self.func(*args, **kwds)
            self.dict[args] = res
            return res


@FrequencyCache
def loss_function(len, NN, NT, TN, TT, log=False):
    if log:
        # total = torch.logsumexp(torch.tensor([NN, NT, TN, TT]), dim=0)
        # NN, NT, TN, TT = NN - total, NT - total, TN - total, TT - total

        if len == 1:
            return 0
        elif len == 2:
            return TT

        loss = torch.full_like(NN, -np.inf)
        for i in range(1, len):
            l = loss_function(i, NN, NT, TN, TT, log=True) + loss_function(
                len - i, NN, NT, TN, TT, log=True
            )
            if i == 1:
                loss = torch.logsumexp(torch.stack([loss, TN + l]), dim=0)
            elif i == len - 1:
                loss = torch.logsumexp(torch.stack([loss, NT + l]), dim=0)
            else:
                loss = torch.logsumexp(torch.stack([loss, NN + l]), dim=0)
    else:
        total = NN + NT + TN + TT
        NN, NT, TN, TT = NN / total, NT / total, TN / total, TT / total

        if len == 1:
            return 1
        elif len == 2:
            return TT

        loss = 0
        for i in range(1, len):
            l = loss_function(i, NN, NT, TN, TT) * loss_function(
                len - i, NN, NT, TN, TT
            )
            if i == 1:
                loss += TN * l
            elif i == len - 1:
                loss += NT * l
            else:
                loss += NN * l

    return loss


def gradient(len, NN, NT, TN, TT, log=False):
    if log:
        total = torch.logsumexp(torch.tensor([NN, NT, TN, TT]), dim=0)
        NN, NT, TN, TT = NN - total, NT - total, TN - total, TT - total

        grad_NN = loss_function(len, NN, NT, TN, TT, log=True) - NN
        grad_NT = loss_function(len, NN, NT, TN, TT, log=True) - NT
        grad_TN = loss_function(len, NN, NT, TN, TT, log=True) - TN
        grad_TT = loss_function(len, NN, NT, TN, TT, log=True) - TT
    else:
        total = NN + NT + TN + TT
        NN, NT, TN, TT = NN / total, NT / total, TN / total, TT / total

        grad_NN = loss_function(len, NN, NT, TN, TT) / NN
        grad_NT = loss_function(len, NN, NT, TN, TT) / NT
        grad_TN = loss_function(len, NN, NT, TN, TT) / TN
        grad_TT = loss_function(len, NN, NT, TN, TT) / TT
    return grad_NN, grad_NT, grad_TN, grad_TT


@FrequencyCache
def freq_tree(l):
    if l == 1:
        return 1
    elif l == 2:
        return 1
    else:
        f = 0
        for i in range(1, l):
            f += freq_tree(i) * freq_tree(l - i)
        return f


@FrequencyCache
def freq_tt(l):
    if l == 2:
        return 1
    else:
        f = 0
        for i in range(2, l):
            f += freq_tree(l - i) * freq_tt(i)
        return 2 * f


@FrequencyCache
def freq_nt(l):
    if l == 2:
        return 0
    else:
        f = 0
        for i in range(2, l):
            f += freq_tree(l - i) * freq_nt(i)
        return 2 * f + freq_tree(l - 1)


def freq_tn(l):
    return freq_nt(l)


@FrequencyCache
def freq_nn(l):
    if l == 2:
        return 0
    else:
        f = 0
        for i in range(2, l):
            f += freq_tree(l - i) * freq_nn(i)
        f = 2 * f
        for i in range(2, l - 1):
            f += freq_tree(i) * freq_tree(l - i)
        return f


def freq_symbols(l):
    return freq_tt(l) + freq_nt(l) + freq_tn(l) + freq_nn(l)


def freq_symbol_each(l):
    return freq_nn(l), freq_nt(l), freq_tn(l), freq_tt(l)


if __name__ == "__main__":
    generate_random_span_by_length(7)
