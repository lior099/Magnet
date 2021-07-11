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

def run_louvain(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    """
    Run our Louvain-like method for community detection, constraining also that the more nodes of the same type there
    are in a community, the worse."""
    num_types = graph.groups_number

    assert len(beta) == num_types, "Beta vector length mismatches the number of types"
    partition = best_partition(graph, resolution=res, beta_penalty=beta)
    partition_to_csv(graph, partition, dump_name)
    # plot_toy_graphs(graph=graph, partition=partition, name="color", graphs_directions=[(0, 1)])
    # if assess:
    #     measure_performance(partition, ground_truth)
    # if draw:
    #     draw_results(graph, partition, dump_name)


def task2(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    np.random.seed(42)
    start = time.time()

    run_louvain(graph, dump_name, res, beta, assess=assess, ground_truth=ground_truth, draw=draw)
    return time.time() - start

def eval_task2(results_files, method):
    results_file = results_files[0]
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
        if method == 'avg_3':
            for c, lst1 in coms.items():
                if len(lst1) == 3 and lst1[0].split("_")[1] == lst1[1].split("_")[1] == lst1[2].split("_")[1]:
                    scores.append(1)
                elif len(lst1) == 3:
                    scores.append(0)
        elif method == 'avg_all':
            for c, lst1 in coms.items():
                if len(lst1) == 3 and lst1[0].split("_")[1] == lst1[1].split("_")[1] == lst1[2].split("_")[1]:
                    scores.append(1)
                else:
                    scores.append(0)
        else:
            raise Exception('method', method, 'not found!')
    return 100 * np.mean(scores)
