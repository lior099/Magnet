import csv
import time
import math

import networkx as nx
import numpy as np

from node2vec import Node2Vec

from MultipartiteCommunityDetection.code.run_louvain import load_graph_from_files
from StaticGraphEmbeddings.evaluation_tasks.calculate_static_embeddings import ogre_static_embeddings

import os
import sys
for root, dirs, files in os.walk(os.path.join('C:/', 'Users', 'Lior', 'Documents', 'University', 'DATA SCIENCE', 'Lab', 'Magnet', 'StaticGraphEmbeddings')):
    sys.path.append(os.path.abspath(root))
    for dir in dirs:
        sys.path.append(os.path.abspath(os.path.join(root, dir)))


from Tests.proj import createKD


def get_closest_neighbors(tree, node, k, node_to_embed, dic):
    closest_neighbors = {}
    dist, ind = tree.query([node_to_embed[node]], k=k)
    for group in ["0", "1", "2"]:
        if group != node.split("_")[0]:
            closest_neighbors[group] = {dic[neighbor]: distance for neighbor, distance in zip(ind[0], dist[0]) if dic[neighbor].split("_")[0] == group}
            closest_neighbors[group] = dict(list(closest_neighbors[group].items())[:5])
    counts = {group: len(nodes) for group, nodes in closest_neighbors.items()}
    return closest_neighbors, counts

def fun_1(x):
    return 1/x

def fun_2(x):
    return 1/(1 + x**2)

def fun_3(x):
    return math.exp(-x)

def fun_4(x):
    return -x

def log_mean(probs, alpha):
    log = [-np.log(p+alpha) for p in probs]
    return np.mean(log)


# TODO create a utils.py and put this function there
def top5_probs_to_csv(probs, filename, start):
    with open(filename, "a", newline='') as file:
        nodes, nodes_probs = [], []
        for group, group_probs in probs.items():
            group_probs = dict(sorted(group_probs.items(), key=lambda item: item[1], reverse=True)[:5])
            nodes += list(group_probs.keys())
            nodes_probs += list(group_probs.values())

        w = csv.writer(file)
        if os.stat(filename).st_size == 0:
            w.writerow(["Source", "Targets and Probabilities"])
        w.writerow([start])
        w.writerow([''] + nodes)
        w.writerow([''] + nodes_probs)

def node2vec_embed(graph_file_names, from_to_ids):
    graph = load_graph_from_files(graph_file_names, from_to_ids)
    # graph = nx.DiGraph()
    # graph.add_weighted_edges_from([[1,2,1], [1,3,2], [1,4,3], [2,1,1], [2,3,2], [2,4,3],[3,2,3], [3,1,2], [1,4,3]])
    # graph.add_weighted_edges_from([[i, j, 1] for i in range(10) for j in range(10) if i != j])
    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    nodes = list(graph.nodes())
    my_dict = {}
    for node in nodes:
        try:
            my_dict.update({node: np.asarray(model.wv.get_vector(node))})
        except KeyError:
            my_dict.update({node: np.asarray(model.wv.get_vector(str(node)))})
    X = np.zeros((len(nodes), 128))
    for i in range(len(nodes)):
        try:
            X[i, :] = np.asarray(model.wv.get_vector(nodes[i]))
        except KeyError:
            X[i, :] = np.asarray(model.wv.get_vector(str(nodes[i])))
    # X is the embedding matrix and projections are the embedding dictionary
    return my_dict, graph
    # return X, my_dict


def task4(graph_file_names, results_file, from_to_ids, embedding=None, epsilon=0.01):
    start = time.time()
    open(results_file, 'w').close()
    distance_functions = [fun_3]  # [fun_1, fun_2, fun_3, fun_4]
    acc_dict = {fun: [] for fun in distance_functions}
    index_count = [0]*6
    # print(directory)
    if embedding == 'ogre':
        z, graph, initial_size, list_initial_proj_nodes = ogre_static_embeddings(graph_file_names, epsilon)
        node_to_embed = z['OGRE + node2vec'].list_dicts_embedding[0]
        # node_to_embed = z['node2vec'][1]
    elif embedding == 'node2vec':
        try:
            node_to_embed, graph = node2vec_embed(graph_file_names, from_to_ids)
        except:
            raise Exception("node2vec crashed...")
    else:
        raise Exception("embedding " + str(embedding) + " was not found")

    tree, dic = createKD(node_to_embed)
    for idx, node in dic.items():
        identity = node.split("_")[1]
        k = 11
        closest_neighbors, counts = get_closest_neighbors(tree, node, k, node_to_embed, dic)
        while min(counts.values()) < 5:
            k += 10
            closest_neighbors, counts = get_closest_neighbors(tree, node, k, node_to_embed, dic)
        # print("Node:",node)
        probs = {}
        for group, nodes in closest_neighbors.items():
            # if identity not in nodes.keys():
            #     index = 5
            # else:
            #     index = list(nodes.keys()).index(identity)
            # index_count[index] += 1
            # print("Found in the index:",index)
            for fun in distance_functions:
                nodes_after_fun = {node: fun(distance) for node, distance in list(nodes.items())[:5]}
                nodes_sum = sum(list(nodes_after_fun.values()))
                if nodes_sum:
                    distances_after_norm = [i/nodes_sum for i in list(nodes_after_fun.values())]
                else:
                    distances_after_norm = [1 if i == min(list(nodes.items())[:5]) else 0 for i in list(nodes.items())[:5]]
                nodes_after_norm = {node: prob for node, prob in zip(nodes_after_fun.keys(), distances_after_norm)}
                # prob = nodes_after_norm.get(identity, 0)
                # acc_dict[fun].append(prob)
            probs[group] = nodes_after_norm
            # print("function 3 got acc", prob)
        top5_probs_to_csv(probs, results_file, node)
    # index_count = [i/sum(index_count) for i in index_count]
    # print(index_count)
    # alpha = 0.003
    # acc = log_mean(acc_dict[fun_3], alpha)
    # print("acc", acc)
    # print("### Graph with ",len(graph.nodes), "nodes acc:",acc)


    # acc = sum(acc_dict[fun_3]) / len(acc_dict[fun_3])
    # alpha_list = [0.00001*2**i for i in range(12)]
    # acc_alpha = [log_mean(acc_dict[fun_3], alpha) for alpha in alpha_list]
    # print("acc_alpha", acc_alpha)
    # return acc
    return time.time() - start


def eval_task4(results_files, method):
    results_file = results_files[0]
    with open(results_file, "r", newline='') as csvfile:
        probs = {}
        datareader = csv.reader(csvfile)
        next(datareader, None)  # skip the headers
        for source in datareader:
            source = source[0]
            group = source.split('_')[0]
            neighbors = next(datareader)[1:]
            neighbors_probs = next(datareader)[1:]
            probs[source] = probs.get(source, {})
            for neighbor, prob in zip(neighbors, neighbors_probs):
                neighbor_group, neighbor = neighbor.split('_')
                if neighbor_group != group:
                    probs[source][neighbor_group] = probs[source].get(neighbor_group, {})
                    probs[source][neighbor_group][neighbor] = float(prob)
        if method == 'avg':
            scores = [neighbors.get(node.split('_')[1], 0) for node, groups in probs.items() for neighbors in groups.values()]
        elif method == 'avg_norm':
            scores = [neighbors.get(node.split('_')[1], 0) / sum(neighbors.values()) for node, groups in probs.items() for neighbors in groups.values()]
        elif method == 'winner':
            scores = [1 if node.split('_')[1] == list(neighbors.keys())[0] else 0 for node, groups in probs.items() for neighbors in groups.values()]
        elif method == 'top5':
            scores = [1 if node.split('_')[1] in list(neighbors.keys())[:5] else 0 for node, groups in probs.items() for neighbors in groups.values()]
        else:
            raise Exception('method', method, 'not found!')
    return 100 * np.mean(scores)


