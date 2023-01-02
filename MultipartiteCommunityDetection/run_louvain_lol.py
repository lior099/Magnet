import os
import csv
import time

import numpy as np

from BipartiteProbabilisticMatching.matching_solutions import plot_toy_graphs
from MultipartiteCommunityDetection.louvain_like_lol import best_partition
from multipartite_lol_graph import MultipartiteLol


def partition_to_csv(graph, partition, filename):
    with open(filename, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["Node", "Type", "Community"])
        for v, c in partition.items():
            w.writerow([v.split("_")[1], np.argmax(graph.return_node_type(v)), c])
        # print("---LOL---")
        # check_accuracy(partition)


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
    return counts_good/float(counts[3]) * 100
    # print("counts",counts)
    # print("counts_good", counts_good)

def run_louvain(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True, greedy=False):
    """
    Run our Louvain-like method for community detection, constraining also that the more nodes of the same type there
    are in a community, the worse."""
    num_types = graph.groups_number

    assert len(beta) == num_types, "Beta vector length mismatches the number of types"
    if greedy:
        partition = greedy_partition(graph)
    else:
        partition = best_partition(graph, resolution=res, beta_penalty=beta)
    partition_to_csv(graph, partition, dump_name)
    # plot_toy_graphs(graph=graph, partition=partition, name="color", graphs_directions=[(0, 1)])
    # if assess:
    #     measure_performance(partition, ground_truth)
    # if draw:
    #     draw_results(graph, partition, dump_name)

def greedy_partition(graph):
    groups, partition = [], {}
    for node in graph.nodes():
        if node[0] != '0':
            continue
        neighbors_shapes = [str(i) for i in range(graph.groups_number) if str(i) != node[0]]
        neighbors = graph.neighbors(node)
        neighbors_dict = {}
        for shape in neighbors_shapes:
            shape_neighbors = [neighbor for neighbor in neighbors[0] if neighbor[0] == shape]
            shape_weights = [weight for neighbor, weight in zip(*neighbors) if neighbor[0] == shape]
            if len(shape_neighbors) > 0:
                neighbors_dict[shape] = (shape_neighbors, shape_weights)

        best_neighbors = []
        for shape_neighbors, shape_weights in neighbors_dict.values():
            best_neighbor_idx = np.argmax(shape_weights)
            best_neighbors.append(shape_neighbors[best_neighbor_idx])
        groups.append([node] + best_neighbors)
    for i, group in enumerate(groups):
        for node in group:
            partition[node] = i
    return partition


def task2(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    np.random.seed(42)
    start = time.time()

    run_louvain(graph, dump_name, res, beta, assess=assess, ground_truth=ground_truth, draw=draw)
    return time.time() - start

def eval_task2(results_files, method, params):
    results_file = results_files[0]
    num_of_groups = params.get('num_of_groups', 3)
    with open(results_file, "r", newline='') as csvfile:
        coms = {}
        datareader = csv.reader(csvfile)
        next(datareader, None)  # skip the headers
        for row in datareader:
            if len(row) != 3:
                continue
            coms[row[2]] = coms.get(row[2], [])
            coms[row[2]].append(row[1]+'_'+row[0])

        scores = []
        # avg between all the groups that have a full match (for example only [0_22, 1_22, 2_22])
        if method == 'avg_full':
            for c, nodes in coms.items():
                nodes_com, nodes_idx = zip(*[node.split("_") for node in nodes])
                # check if nodes_idx has only one unique value, it means the community is good
                if len(nodes) == num_of_groups and len(set(nodes_idx)) == 1:
                    scores.append(1)
                elif len(nodes) == num_of_groups:
                    scores.append(0)
        # avg between all the groups (for example [0_22, 1_22, 2_22], [0_22, 2_22])
        elif method == 'avg_all':
            for c, nodes in coms.items():
                nodes_com, nodes_idx = zip(*[node.split("_") for node in nodes])
                # check if nodes_idx has only one unique value, it means the community is good
                if len(nodes) == num_of_groups and len(set(nodes_idx)) == 1:
                    scores.append(1)
                else:
                    scores.append(0)
        else:
            raise Exception('method', method, 'not found!')
    return 100 * np.mean(scores)
