import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from MultipartiteCommunityDetection.code.louvain_like import best_partition
import time


def load_graph_from_files(filenames, identities, has_title=True, cutoff=0.):
    """
    Build a directed graph appropriate for our Louvain algorithm.
    :param filenames: list of file names of the edges (source, target, proability) from the first stage.
    :param identities: list of tuples [(source_type, target_type) for file in filenames],
    determining which types of vertices are represented in each file. source_type and target_type can be any object,
    but it is recommended to input them as strings.
    :param has_title: A boolean indicating whether the files start with a title row
    :param cutoff: A float that edges with weights smaller than which are removed from the graph.
    :return: The final directed graph.
    """
    graph = nx.DiGraph()
    all_types = set(itertools.chain.from_iterable(identities))
    type_to_idx = {identity: i for i, identity in enumerate(all_types)}  # Save this matching?
    for file, (id_source, id_target) in zip(filenames, identities):
        with open(file, "r") as f:
            if has_title:
                next(f)  # First line has titles
            for line in f:
                source, target, weight = line.strip().split(",")
                weight = float(weight)
                if f"{id_source}_{source}" not in graph:
                    graph.add_node(f"{id_source}_{source}",
                                   type=[1 if type_to_idx[id_source] == i else 0 for i in range(len(type_to_idx))])
                if f"{id_target}_{target}" not in graph:
                    graph.add_node(f"{id_target}_{target}",
                                   type=[1 if type_to_idx[id_target] == i else 0 for i in range(len(type_to_idx))])
                if weight > cutoff:
                    graph.add_edge(f"{id_source}_{source}", f"{id_target}_{target}", weight=weight)
    return graph


def draw_results(graph, partition, filename):
    # Not very recommended for graphs with more than 200 nodes.
    num_types = len(graph.node[list(graph.nodes())[0]]['type'])
    count_per_shape = {i: len([node for node, data in graph.nodes(data=True) if data['type'][i]])
                       for i in range(num_types)}
    indices = [0] * num_types
    pos = {}
    for node, data in graph.nodes(data=True):
        node_type = np.argmax(data['type'])
        pos[node] = np.array([10 * node_type / float(num_types),
                              10 * indices[int(node_type)] / max(1., count_per_shape[int(node_type)] - 1)])
        indices[int(node_type)] += 1
    possible_shapes = {0: "o", 1: "d", 2: "s", 3: "^", 4: "X", 5: "v", 6: "p", 7: "P", 8: "*", 9: "h"}
    max_edge_weight = max([w['weight'] for _, _, w in graph.edges(data=True)])
    edge_colors = [float(w['weight']) / max_edge_weight for _, _, w in graph.edges(data=True)]
    node_colors = {c: plt.get_cmap('hsv')(float(c) / len(set(partition.values()))) for c in set(partition.values())}

    for i in range(num_types):
        nodes_to_draw = [node for node, data in graph.nodes(data=True) if data['type'][i]]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes_to_draw, node_shape=possible_shapes[i % 10],
                               node_color=[node_colors[partition[v]] for v in nodes_to_draw], label=nodes_to_draw)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=4, edge_color=edge_colors,
                           edge_cmap=plt.cm.winter, width=0.1)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"{filename}.png"))


def partition_to_csv(graph, partition, filename):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"{filename}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["Node", "Type", "Community"])
        for v, c in partition.items():
            w.writerow([v.split("_")[1], np.argmax(graph.nodes[v]['type']), c])
        print("---NETWORKX---")
        check_accuracy(partition)


def check_accuracy(partition):
    coms = {}
    lst = []
    print("len(partition)",len(partition))
    for v, c in partition.items():
        lst.append(v)
        if c not in coms.keys():
            coms[c] = [v]
        else:
            coms[c].append(v)
    counts = {1: 0, 2: 0, 3: 0}
    counts_good = 0
    for c, lst1 in coms.items():
        # print(len(lst), lst)
        counts[len(lst1)] += 1
        if len(lst1) == 3 and lst1[0].split("_")[1] == lst1[1].split("_")[1] == lst1[2].split("_")[1]:
            counts_good += 1
    print("counts",counts)
    print("counts_good", counts_good)


def measure_performance(partition, ground_truth):
    """Return a score indicating how accurate the algorithm is by the partition and the ground truth sets."""
    assert ground_truth is not None, 'For measuring the performance, a ground truth is required'
    node_to_type = {n: n.split("_")[0] for n in partition.keys()}
    num_types = len(dict.fromkeys(node_to_type.values()))
    values = np.arange(max(partition.values()) + 1)  # communities have values 0, 1, ..., number_of_communities-1
    com_to_counts = {value: [0] * num_types for value in values}
    com_to_nodes = {value: [] for value in values}
    for node, com in partition.items():
        com_to_counts[com][int(node_to_type[node])] += 1
        com_to_nodes[com].append(node)
    counts_to_com = {}
    for com, counts in com_to_counts.items():
        counts = "_".join(map(str, counts))
        if counts in counts_to_com:
            counts_to_com[counts].append(com)
        else:
            counts_to_com[counts] = [com]

    pure_communities = counts_to_com.get("_".join(["1"] * num_types), [])
    pure_communities_ratio = len(pure_communities) / len(values)
    print(f"Fraction of communities with exactly one node of each type: {pure_communities_ratio}")

    pure_community_nodes = [com_to_nodes[com] for com in pure_communities]
    ground_truth_strings = set("_".join(sorted(c)) for c in ground_truth)
    pure_community_strings = set("_".join(sorted(c)) for c in pure_community_nodes)
    caught_communities_ratio = len(pure_community_strings.intersection(ground_truth_strings)) / float(len(ground_truth))
    print(f"Fraction of communities exactly caught: {caught_communities_ratio}")


def run_louvain(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    """
    Run our Louvain-like method for community detection, constraining also that the more nodes of the same type there
    are in a community, the worse."""
    num_types = len(graph.nodes[list(graph.nodes())[0]]['type'])
    assert len(beta) == num_types, "Beta vector length mismatches the number of types"
    partition = best_partition(graph, resolution=res, beta_penalty=beta)
    partition_to_csv(graph, partition, dump_name)
    if assess:
        measure_performance(partition, ground_truth)
    if draw:
        draw_results(graph, partition, dump_name)


def load_ground_truths(filename):
    df = pd.read_csv(filename, header=None).T
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{col}_{x}")
    ground_truth = [df.iloc[i].tolist() for i in range(df.shape[0])]
    return ground_truth


def task2(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    np.random.seed(42)
    run_louvain(graph, dump_name, res, beta, assess=False, ground_truth=None, draw=False)

