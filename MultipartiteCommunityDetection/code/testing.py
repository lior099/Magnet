import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from MultipartiteCommunityDetection.code.louvain_like import best_partition
from lol_graph import LolGraph
from multipartite_lol_graph import MultipartiteLol




def load_graph_from_files2(filenames, identities, has_title=True, cutoff=0.):
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
    graph = nx.Graph()
    all_types = set(itertools.chain.from_iterable(identities))
    type_to_idx = {identity: i for i, identity in enumerate(all_types)}  # Save this matching?
    for file, (id_source, id_target) in zip(filenames, identities):
        with open(file, "r") as f:
            if has_title:
                next(f)  # First line has titles
            for line in f:
                source, target, weight = line.strip().split(",")
                weight = float(weight)
                if f"{source}" not in graph:
                    graph.add_node(f"{source}",
                                   type=[1 if type_to_idx[id_source] == i else 0 for i in range(len(type_to_idx))])
                if f"{target}" not in graph:
                    graph.add_node(f"{target}",
                                   type=[1 if type_to_idx[id_target] == i else 0 for i in range(len(type_to_idx))])
                if weight > cutoff:
                    # graph.add_edge(f"{source}", f"{target}", weight=weight)
                    graph.add_edge(f"{source}", f"{target}")
    return graph


if __name__ == '__main__':
    file = "toy_network_2"
    rootDir = os.path.join("..", "..", "PathwayProbabilitiesCalculation", "data", file)
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                       os.walk(rootDir) for file in filenames]
    graph_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]  #[(0, i) for i in range(1, 4)]
    graph_filenames.sort()
    gr = load_graph_from_files2(graph_filenames[0:1], graph_ids[0:1], has_title=True)


    gr1 = LolGraph(directed=False, weighted=False)
    rootDir = os.path.join("..", "..", "PathwayProbabilitiesCalculation", "data", file)
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                       os.walk(rootDir) for file in filenames]
    graph_filenames.sort()
    # gr1.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], True)
    gr1.convert_with_csv(graph_filenames[0:1], True)
    # list_of_list_graph.set_nodes_type_dict()

    node1 = "2"
    node2 = "6"

    print("is_directed", str(gr1.is_directed()) == str(gr.is_directed()))
    # print("is_weighted", gr1.is_weighted())
    print("number_of_edges", str(gr1.number_of_edges()) == str(float(gr.number_of_edges())))
    print("number_of_nodes", str(gr1.number_of_nodes()) == str(gr.number_of_nodes()))
    print("out_degree", str(float(gr1.out_degree(node1))) == str(gr.degree(node1, weight="weight")))
    # print("in_degree", str(float(gr1.in_degree(node1))) == str(gr.in_degree(node1, weight="weight")))
    # print("predecessors", str(gr1.predecessors(node1)) == str(list(gr.predecessors(node1))))
    print("nodes", str(gr1.nodes()) == str(gr.nodes()))
    print("size", str(float(gr1.size())) == str(gr.size(weight="weight")))
    print("get_edge_data", str(gr1.get_edge_data(node1, node2)) == str(gr.get_edge_data(node1, node2)) or gr.get_edge_data(node1, node2) == None)
    print("neighbors", str(gr1.neighbors(node1)[0]) == str(list(gr.neighbors(node1))))
    # print("graph_adjacency", str(gr1.graph_adjacency()) == str(gr.adj))
    # print("edges", str(gr1.edges()) == str([[a,b,c["weight"]] for a,b,c in gr.edges(data=True)]))


    # j = 0
    # for i,b in zip(list(str(gr1.edges())), list(str([[a,b,c["weight"]] for a,b,c in gr.edges(data=True)]))):
    #     print(j,i==b)
    #     j+=1
    # print("is_edge_between_nodes", gr1.is_edge_between_nodes(node1, node2))
    # print(list_of_list_graph.add_edges())
    # print(list_of_list_graph.swap_edge())
    print("------------------LOL----------------\n-------------------------------------")
    import time
    start = time.time()
    print("is_directed", gr1.is_directed())
    end = time.time()
    print("time:",end-start)
    start = time.time()
    # print("is_weighted", gr1.is_weighted())
    print("number_of_edges", gr1.number_of_edges())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("number_of_nodes", gr1.number_of_nodes())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("out_degree", gr1.out_degree(node1))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("in_degree", gr1.in_degree(node1))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("predecessors", len(gr1.predecessors(node1)))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("nodes", len(gr1.nodes()))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("edges", len(gr1.edges()))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("is_edge_between_nodes", gr1.is_edge_between_nodes(node1, node2))
    print("size", gr1.size())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("get_edge_data", gr1.get_edge_data(node1, node2))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("neighbors", len(gr1.neighbors(node1)))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("graph_adjacency", len(gr1.graph_adjacency()))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print(list_of_list_graph.add_edges())
    # print(list_of_list_graph.swap_edge())
    print("------------NETWORKX-----------------\n-------------------------------------")
    print("is_directed", gr.is_directed())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("is_weighted",gr.is_weighted())
    print("number_of_edges", gr.number_of_edges())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("number_of_nodes", gr.number_of_nodes())
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("out_degree", gr.degree(node1, weight="weight"))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("in_degree", gr.in_degree(node1, weight="weight"))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("predecessors", len(list(gr.predecessors(node1))))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("nodes", len(gr.nodes()))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("edges", str([[a,b,c["weight"]] for a,b,c in gr.edges(data=True)]))
    print("edges", len(gr1.edges()))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    # print("is_edge_between_nodes",gr.is_edge_between_nodes())
    print("size", gr.size(weight="weight"))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("get_edge_data", gr.get_edge_data(node1, node2))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("neighbors", len(list(gr.neighbors(node1))))
    end = time.time()
    print("time:", end - start)
    start = time.time()
    print("graph_adjacency", len(gr.adj))
    end = time.time()
    print("time:", end - start)
    start = time.time()
