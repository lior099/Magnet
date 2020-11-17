import time
import networkx as nx
from queue import Queue
import copy
import os
import pandas as pd

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

    edges_dict = dict(graph.adj)
    probs = {point: {i: 0 for i in range(1, k + 1)} for point in edges_dict.keys()}
    probs[starting_point] = {0: 1}

    distance_from_source, layers_dict = bfs(graph, starting_point, k)
    attempt_temp_probs = dict()
    deal_later_set = list()
    # iterate by layer
    print("START")
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
            if group_sums[group] is not 0:
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
def bfs(graph, starting_point, k):
    # dictionary holding the neighbors of each node
    edges_dict = dict(graph.adj)
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
    layers_dict = {i: [neighbour for neighbour, dis in distance_from_start.items() if dis == i] for i in range(k + 1)}
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
        multipartite_graph.add_weighted_edges_from(list(bipartite_graph.values))

    print("return")
    return multipartite_graph


def main():
    for network in range(8, 9):
        graph_filenames = []
        for g in range(1, 4):
            graph_filenames.append(
                os.path.join(f"yoram_network_{network}", f"yoram_network_{network}_graph_{g}_01.csv"))
            graph_filenames.append(
                os.path.join(f"yoram_network_{network}", f"yoram_network_{network}_graph_{g}_10.csv"))

        multipartite_graph = load_graph(graph_filenames)
        print("Run number {}".format(network))
        print("The number of nodes are {}, and the number of edges are {}".format(multipartite_graph.number_of_nodes(),
                                                                                  multipartite_graph.number_of_edges()))
        probs = iterate_by_layers(multipartite_graph, 4, '2a')
        print(probs)
        print(normalize_probs_matrix(probs))

    # t = time.time()
    # multipartite_graph = load_graph(graph_filenames)
    # estimate_time = time.time() - t
    # iterate_by_layers(multipartite_graph, 4, '1a')
    # print("Run number {} takes {}".format(i, estimate_time))
    # print("The number of nodes are {}, and the number of edges are {}".format(multipartite_graph.number_of_nodes(),
    #                                                                           multipartite_graph.number_of_edges()))

    # print(iterate_by_layers(multipartite_graph, 4, '2a'))

    # print(multipartite_graph.edges)

    # print("The number of nodes are {}, and the number of edges are {}".format(multipartite_graph.number_of_nodes(),
    # multipartite_graph.number_of_edges()))
    ''''
        QG = nx.DiGraph()
        QG.add_nodes_from(['1', '2'], agentType='alpha')
        QG.add_nodes_from(['a', 'b', 'c', 'd'], agentType='beta')
        QG.add_nodes_from(['A', 'B'], agentType='gamma')
        QG.add_nodes_from(['3', '4', '5'], agentType='delta')
        myEdges1 = [('1', 'a', 1.3), ('1', 'c', 1), ('1', 'd', 1), ('1', 'A', 1), ('2', 'd', 1),
                    ('a', 'A', 1), ('b', 'A', 0.5), ('b', 'B', 0.5), ('c', 'B', 1), ('d', '5', 0.6), ('d', '3', 0.4),
                    ('A', '3', 1), ('B', '4', 0.7), ('B', '4', 0.7), ('B', '5', 0.3)]
        QG.add_weighted_edges_from(myEdges1)
        multipartite_graph = QG
        '''''
    '''''
    QG2 = nx.DiGraph()
    QG2.add_nodes_from(['0'], agentType='alpha')
    QG2.add_nodes_from(['1', '2', '3'], agentType='beta')
    QG2.add_nodes_from(['4', '5'], agentType='gamma')
    myEdges2 = [('0', '1', 0.2), ('0', '2', 0.2), ('0', '3', 0.6),
                ('1', '2', 0.1), ('1', '3', 0.1), ('1', '4', 0.3), ('1', '5', 0.5),
                ('2', '1', 0.1), ('2', '3', 0.9),
                ('3', '1', 0.8), ('3', '2', 0.2),
                ('4', '5', 1)]
    QG2.add_weighted_edges_from(myEdges2)
    '''''


main()