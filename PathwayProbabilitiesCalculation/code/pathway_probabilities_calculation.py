import time
import networkx as nx
from queue import Queue
import copy
import os
import pandas as pd

from lol_graph import lol_graph

'''''
def get_probs(graph, k, starting_point):
    edges_dict = dict(graph.adj)
    probs = {point: {i: 0 for i in range(1, k + 1)} for point in edges_dict.keys()}
    probs[starting_point] = {0: 1}
    for source, neighbors in edges_dict.items():
        for point, weight_dict in neighbors.items():
            weight = weight_dict['weight']
            for steps, prob in probs[source].items():
                if steps is not k:
                    probs[point][steps + 1] += prob * weight
    print(probs)
'''''


# The function calculates the probability to get from start to all nodes.
# Input:
#      graph
#      start - the node we travel from.
#      k - the number of maximum allowed steps.
# Output:
#      probs - dictionary mapping between each node to a
#              dictionary holding the probability to get to this node from starting_point.
def iterate_by_layers(graph, k, starting_point):
    starting_group = starting_point[-1]
    # edges_dict = dict(graph.adj)
    edges_dict = graph.graph_adjacency()
    probs = {point: {i: 0 for i in range(1, k + 1)} for point in edges_dict.keys()}
    probs[starting_point] = {0: 1}

    distance_from_source, layers_dict = bfs(edges_dict, starting_point, k)
    attempt_temp_probs = dict()
    deal_later_set = list()
    # iterate by layer
    for layer, vertices_set in layers_dict.items():
        # temp_probs = copy.deepcopy(probs)
        attempt_temp_probs = copy.deepcopy({vertex: probs[vertex] for vertex in vertices_set})
        # iterate the vertices in the current Layer
        for vertex in vertices_set:
            current_vertex_neighbors = edges_dict[vertex]
            # iterate the neighbours of current Vertex
            for neighbour, weight_dict in current_vertex_neighbors.items():
                # ignore the neighbours in lower layer
                if distance_from_source[neighbour] < distance_from_source[vertex] or neighbour[-1] == starting_group:
                    continue
                # deal first with the neighbours in the same layer as Vertex
                if distance_from_source[neighbour] == distance_from_source[vertex]:
                    weight = weight_dict['weight']
                    for steps, prob in probs[vertex].items():
                        if steps is not k:
                            # temp_probs[neighbour][steps + 1] = \
                            #     round(temp_probs[neighbour][steps + 1] + prob * weight, 10)
                            attempt_temp_probs[neighbour][steps + 1] = \
                                round(attempt_temp_probs[neighbour][steps + 1] + prob * weight, 10)

                # deal with the neighbours in greater layer only after we finish dealing with
                # the neighbours in the same layer
                if distance_from_source[neighbour] > distance_from_source[vertex]:
                    deal_later_set.append((vertex, neighbour))
        # probs = temp_probs
        # copy back to probs dict
        for vertex in attempt_temp_probs.keys():
            probs[vertex] = attempt_temp_probs[vertex]
        deal_later_vertex(deal_later_set, k, probs, edges_dict)
    return probs


def normalize_probs_matrix(probs):
    vertex_sums = dict()
    for vertex, probs_dict in probs.items():
        group = vertex[-1]
        if group not in vertex_sums.keys():
            vertex_sums[group] = dict()
        vertex_sums[group][vertex] = sum(probs_dict.values())
    group_sums = {group: sum(group_vertex_sums.values()) for group, group_vertex_sums in vertex_sums.items()}
    for group, group_vertex_sums in vertex_sums.items():
        for vertex in group_vertex_sums.keys():
            if group_sums[group] != 0:
                group_vertex_sums[vertex] /= group_sums[group]
    return vertex_sums


# completeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
# Input:
#      probs - dictionary mapping between each node to a
#              dictionary holding the probability to get to this node from starting_point.
#      deal_later_set - a list with vertices in greater layer that the current layer being iterated.
#      edges_dict - dictionary holding all nodes and their neighbours.
#      k - the number of maximum allowed steps/
def deal_later_vertex(deal_later_set, k, probs, edges_dict):
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
#      layers_dict - dictionary mapping each layer to all its vertices
def bfs(edges_dict, starting_point, k):
    # dictionary holding the distance of each node from Start
    distance_from_start = {starting_point: 0}
    # dictionary mapping each layer to all its vertices
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
                # another way to initialize the dictionary that maps between each layer to all its vertices
                if distance_from_start[neighbour] not in layers_dict:
                    layers_dict[distance_from_start[neighbour]] = list()
                layers_dict[distance_from_start[neighbour]].append(neighbour)
                '''''
    # initialize the dictionary that maps between each layer to all its vertices using One Liner
    layers_dict = {i: [neighbour for neighbour, dis in distance_from_start.items() if dis == i] for i in range(k+1)}
    return distance_from_start, layers_dict


def load_graph(files_array):
    multipartite_graph = nx.DiGraph()

    bipartite_graph_array = [pd.read_csv(file_name) for file_name in files_array]
    # add vertices
    bipartite_graph_array[0].iloc[:, 0] = [str(node) + 'a' for node in bipartite_graph_array[0].iloc[:, 0]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[0].iloc[:, 0], bipartite=0)
    bipartite_graph_array[0].iloc[:, 1] = [str(node) + 'b' for node in bipartite_graph_array[0].iloc[:, 1]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[0].iloc[:, 1], bipartite=1)

    bipartite_graph_array[1].iloc[:, 0] = [str(node) + 'b' for node in bipartite_graph_array[1].iloc[:, 0]]
    bipartite_graph_array[1].iloc[:, 1] = [str(node) + 'a' for node in bipartite_graph_array[1].iloc[:, 1]]

    bipartite_graph_array[2].iloc[:, 0] = [str(node) + 'b' for node in bipartite_graph_array[2].iloc[:, 0]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[2].iloc[:, 0], bipartite=1)
    bipartite_graph_array[2].iloc[:, 1] = [str(node) + 'c' for node in bipartite_graph_array[2].iloc[:, 1]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[2].iloc[:, 1], bipartite=2)

    bipartite_graph_array[3].iloc[:, 0] = [str(node) + 'c' for node in bipartite_graph_array[3].iloc[:, 0]]
    bipartite_graph_array[3].iloc[:, 1] = [str(node) + 'b' for node in bipartite_graph_array[3].iloc[:, 1]]

    bipartite_graph_array[4].iloc[:, 0] = [str(node) + 'c' for node in bipartite_graph_array[4].iloc[:, 0]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[4].iloc[:, 0], bipartite=2)
    bipartite_graph_array[4].iloc[:, 1] = [str(node) + 'a' for node in bipartite_graph_array[4].iloc[:, 1]]
    multipartite_graph.add_nodes_from(bipartite_graph_array[4].iloc[:, 1], bipartite=0)

    bipartite_graph_array[5].iloc[:, 0] = [str(node) + 'a' for node in bipartite_graph_array[5].iloc[:, 0]]
    bipartite_graph_array[5].iloc[:, 1] = [str(node) + 'c' for node in bipartite_graph_array[5].iloc[:, 1]]

    # add edges
    for bipartite_graph in bipartite_graph_array:
        for ad in range(0, bipartite_graph.shape[0]):
            multipartite_graph.add_weighted_edges_from([tuple(bipartite_graph.values[ad])])

    return multipartite_graph


def task3(limit_of_steps, starting_point, graph_files_name, from_to_groups):
    list_of_list_graph = lol_graph()
    list_of_list_graph.convert_with_csv(graph_files_name, from_to_groups, directed=True, weighted=True)

    probs = iterate_by_layers(list_of_list_graph, limit_of_steps, starting_point)
    passway_probability = normalize_probs_matrix(probs)
    return passway_probability
