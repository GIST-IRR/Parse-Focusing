from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import networkx as nx


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
