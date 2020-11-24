import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from MultipartiteCommunityDetection.code.louvain_like_lol import best_partition
from multipartite_lol_graph import Multipartite_Lol


def partition_to_csv(graph, partition, filename):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", f"{filename}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["Node", "Type", "Community"])
        for v, c in partition.items():
            w.writerow([v.split("_")[1], np.argmax(graph.return_node_type(v)), c])


def run_louvain(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    """
    Run our Louvain-like method for community detection, constraining also that the more nodes of the same type there
    are in a community, the worse."""
    num_types = graph.groups_number

    assert len(beta) == num_types, "Beta vector length mismatches the number of types"
    partition = best_partition(graph, resolution=res, beta_penalty=beta)
    partition_to_csv(graph, partition, dump_name)
    # if assess:
    #     measure_performance(partition, ground_truth)
    # if draw:
    #     draw_results(graph, partition, dump_name)


def task2(graph, dump_name, res, beta, assess=True, ground_truth=None, draw=True):
    np.random.seed(42)
    run_louvain(graph, dump_name, res, beta, assess=False, ground_truth=None, draw=False)


if __name__ == '__main__':
    np.random.seed(42)
    rootDir = os.path.join("..", "..", "MultipartiteCommunityDetection", "data", "yoram_network_1")
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                       os.walk(rootDir) for file in filenames]
    list_of_list_graph = Multipartite_Lol()
    graph_filenames.sort()
    list_of_list_graph.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], True, True)
    list_of_list_graph.set_nodes_type_dict()
    run_louvain(list_of_list_graph, "yoram_network_1_lol", 0., [10., 10., 10.],
                assess=False, ground_truth=None, draw=False)
