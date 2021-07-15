import itertools
import time
from queue import Queue
import copy
import os
from random import sample

from multipartite_lol_graph import MultipartiteLol
from lol_graph import *

# The function calculates the probability to get from start to all nodes.
# Input:
#      graph
#      start - the node we travel from.
#      k - the number of maximum allowed steps.
# Output:
#      probs - dictionary mapping between each node to a
#              dictionary holding the probability to get to this node from starting_point.
def iterate_by_layers(graph, k, starting_point):
    starting_group = starting_point[0]
    # edges_dict = dict(graph.adj)
    edges_dict = graph.graph_adjacency()
    probs = {point: {i: 0 for i in range(1, k + 1)} for point in edges_dict.keys()}
    probs[starting_point] = {0: 1}

    distance_from_source, layers_dict = bfs(edges_dict, starting_point, k)
    attempt_temp_probs = dict()
    deal_later_set = list()
    # iterate by layer
    for layer, nodes_set in layers_dict.items():
        # temp_probs = copy.deepcopy(probs)
        attempt_temp_probs = copy.deepcopy({node: probs[node] for node in nodes_set})
        # iterate the nodes in the current Layer
        for node in nodes_set:
            current_node_neighbors = edges_dict[node]
            # iterate the neighbours of current node
            for neighbour, weight_dict in current_node_neighbors.items():
                # ignore the neighbours in lower layer
                if distance_from_source[neighbour] < distance_from_source[node] or neighbour[0] == starting_group:
                    continue
                # deal first with the neighbours in the same layer as node
                if distance_from_source[neighbour] == distance_from_source[node]:
                    weight = weight_dict['weight']
                    for steps, prob in probs[node].items():
                        if steps is not k:
                            # temp_probs[neighbour][steps + 1] = \
                            #     round(temp_probs[neighbour][steps + 1] + prob * weight, 10)
                            attempt_temp_probs[neighbour][steps + 1] = \
                                round(attempt_temp_probs[neighbour][steps + 1] + prob * weight, 10)

                # deal with the neighbours in greater layer only after we finish dealing with
                # the neighbours in the same layer
                if distance_from_source[neighbour] > distance_from_source[node]:
                    deal_later_set.append((node, neighbour))
        # probs = temp_probs
        # copy back to probs dict
        for node in attempt_temp_probs.keys():
            probs[node] = attempt_temp_probs[node]
        deal_later_node(deal_later_set, k, probs, edges_dict)
    return probs


def normalize_probs_matrix(probs):
    nodes_sums = dict()
    for node, probs_dict in probs.items():
        group = node[0]
        if group not in nodes_sums.keys():
            nodes_sums[group] = dict()
        nodes_sums[group][node] = sum(probs_dict.values())
    group_sums = {group: sum(group_node_sums.values()) for group, group_node_sums in nodes_sums.items()}
    for group, group_node_sums in nodes_sums.items():
        for node in group_node_sums.keys():
            if group_sums[group] != 0:
                group_node_sums[node] /= group_sums[group]
    return nodes_sums


# Input:
#      probs - dictionary mapping between each node to a
#              dictionary holding the probability to get to this node from starting_point.
#      deal_later_set - a list with nodes in greater layer that the current layer being iterated.
#      edges_dict - dictionary holding all nodes and their neighbours.
#      k - the number of maximum allowed steps/
def deal_later_node(deal_later_set, k, probs, edges_dict):
    for source, target in deal_later_set:
        weight = edges_dict[source][target]['weight']
        for steps, prob in probs[source].items():
            if steps is not k:
                probs[target][steps + 1] = round(probs[target][steps + 1] + prob * weight, 10)
    deal_later_set.clear()


# The function implements bfs on (connected component) graph.
# Input:
#      graph
#      start - the node we travel from
#      k - the number of maximum allowed steps
# Output:
#      distance_from_start - dictionary holding the distance of each node from Start
#      layers_dict - dictionary mapping each layer to all its nodes
def bfs(edges_dict, starting_point, k):
    # dictionary holding the distance of each node from Start
    distance_from_start = {starting_point: 0}
    # dictionary mapping each layer to all its nodes
    layers_dict = {0: starting_point}

    q = Queue(maxsize=0)
    q.put(starting_point)

    while not q.empty():
        node = q.get()
        neighbours = edges_dict[node].keys()
        # iterate all the neighbors of current node
        for neighbour in neighbours:
            # if the neighbour has never been visited
            if neighbour not in distance_from_start:
                q.put(neighbour)
                # update the neighbour's distance from Start to be its parents' distance plus one
                distance_from_start[neighbour] = distance_from_start[node] + 1

                '''''
                # another way to initialize the dictionary that maps between each layer to all its nodes
                if distance_from_start[neighbour] not in layers_dict:
                    layers_dict[distance_from_start[neighbour]] = list()
                layers_dict[distance_from_start[neighbour]].append(neighbour)
                '''''
    # initialize the dictionary that maps between each layer to all its nodes using One Liner
    layers_dict = {i: [neighbour for neighbour, dis in distance_from_start.items() if dis == i] for i in range(k+1)}
    return distance_from_start, layers_dict


def probs_to_csv(probs, filename, start):
    with open(filename, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target", "Probability"])
        for group, group_probs in probs.items():
            for target, probability in group_probs.items():
                w.writerow([start, target, probability])


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


def task3(limit_of_steps, starting_points, graph_files_name, from_to_groups, destination):
    start = time.time()
    open(destination, 'w').close()
    list_of_list_graph = MultipartiteLol()
    list_of_list_graph.convert_with_csv(graph_files_name, from_to_groups)
    list_of_list_graph.set_nodes_type_dict()
    nodes = list_of_list_graph.nodes()
    sampled_nodes = sample(nodes, starting_points)
    for start_point in sampled_nodes:
        probs = iterate_by_layers(list_of_list_graph, limit_of_steps, start_point)
        passway_probability = normalize_probs_matrix(probs)
        top5_probs_to_csv(passway_probability, destination, start_point)
    return time.time() - start

def eval_task3(results_files, method, params):
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
            scores = [neighbors.get(node.split('_')[1], 0) / sum(neighbors.values()) if sum(neighbors.values()) else 0 for node, groups in probs.items() for neighbors in groups.values()]
        elif method == 'winner':
            scores = [1 if node.split('_')[1] == list(neighbors.keys())[0] else 0 for node, groups in probs.items() for neighbors in groups.values()]
        elif method == 'top5':
            scores = [1 if node.split('_')[1] in list(neighbors.keys())[:5] else 0 for node, groups in probs.items() for neighbors in groups.values()]
        else:
            raise Exception('method', method, 'not found!')
    return 100 * np.mean(scores)


# def task3_for_acc(limit_of_steps, starting_point, graph, from_to_groups, destination):
#     probs = iterate_by_layers(graph, limit_of_steps, starting_point)
#     passway_probability = normalize_probs_matrix(probs)
#     # probs_to_csv(passway_probability, destination, starting_point)
#     return passway_probability
