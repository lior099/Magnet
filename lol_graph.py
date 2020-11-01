import os
import numpy as np
from collections import OrderedDict
import csv


class lol_graph:

    def __init__(self):
        self._index_list = [0]
        self._neighbors_list = []
        self._weights_list = []
        self._map_vertex_to_number = OrderedDict()
        self._map_number_to_vertex = OrderedDict()
        self.directed = False
        self.weighted = False

    # input: csv file containing edges list, in the form of [[5,1],[2,3],[5,3],[4,5]]
    def convert_with_csv(self, files_name, array_of_groups, directed=False, weighted=False):
        self._map_vertex_to_number = OrderedDict()
        group_names_dict = dict()
        ch = 'a'

        for tuple in array_of_groups:
            if tuple[0] not in group_names_dict.keys():
                group_names_dict[tuple[0]] = ch
                ch = chr(ord(ch) + 1)
            if tuple[1] not in group_names_dict.keys():
                group_names_dict[tuple[1]] = ch
                ch = chr(ord(ch) + 1)


        list_of_list = []
        for i in range(len(files_name)):
            file = files_name[i]
            with open(file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                next(datareader, None)  # skip the headers
                for edge in datareader:
                    named_edge = [edge[0] + group_names_dict[array_of_groups[i][0]],
                                  edge[1] + group_names_dict[array_of_groups[i][1]]]
                    if weighted:
                        named_edge.append(float(edge[2]))
                    list_of_list.append(named_edge)
                csvfile.close()
        self.convert(list_of_list, directed, weighted)

    # input: np array of edges, in the form of np array [[5,1,0.1],[2,3,3],[5,3,0.2],[4,5,9]]
    def convert(self, graph, directed=False, weighted=False):
        self._map_vertex_to_number = OrderedDict()
        self.directed = directed
        self.weighted = weighted
        free = 0
        '''create dictionary to self._map_vertex_to_number from edges to our new numbering'''

        for edge in graph:
            if self._map_vertex_to_number.get(edge[0], None) is None:
                self._map_vertex_to_number[edge[0]] = free
                free += 1
            if self._map_vertex_to_number.get(edge[1], None) is None:
                self._map_vertex_to_number[edge[1]] = free
                free += 1

        '''create the opposite dictionary'''
        self._map_number_to_vertex = OrderedDict((y, x) for x, y in self._map_vertex_to_number.items())

        d = OrderedDict()
        '''starting to create the index list. Unordered is important'''
        for idx, edge in enumerate(graph):
            d[self._map_vertex_to_number[edge[0]]] = d.get(self._map_vertex_to_number[edge[0]], 0) + 1
            if not directed:
                d[self._map_vertex_to_number[edge[1]]] = d.get(self._map_vertex_to_number[edge[1]], 0) + 1
            elif self._map_vertex_to_number[edge[1]] not in d.keys():
                d[self._map_vertex_to_number[edge[1]]] = 0

        # list = [0]
        '''transfer the dictionary to list'''
        for j in range(1, len(d.keys()) + 1):
            self._index_list.append(self._index_list[j - 1] + d.get(j - 1, 0))

        '''create the second list'''
        if directed:
            self._neighbors_list = [-1] * len(graph)
        else:
            self._neighbors_list = [-1] * len(graph) * 2
        if weighted:
            self._weights_list = [0] * len(graph)

        space = OrderedDict((x, -1) for x in self._map_number_to_vertex.keys())
        for idx, edge in enumerate(graph):
            left = self._map_vertex_to_number[edge[0]]
            right = self._map_vertex_to_number[edge[1]]
            if weighted:
                weight = float(edge[2])

            if space[left] != -1:
                space[left] += 1
                i = space[left]
            else:
                i = self._index_list[left]
                space[left] = i
            self._neighbors_list[i] = right
            if weighted:
                self._weights_list[i] = weight

            if not directed:
                if space[right] != -1:
                    space[right] += 1
                    i = space[right]
                else:
                    i = self._index_list[right]
                    space[right] = i
                self._neighbors_list[i] = left
                if weighted:
                    self._weights_list[i] = weight
        self.sort_neighbors()
        print()

    # convert back to [[5,1,0.1],[2,3,3],[5,3,0.2],[4,5,9]] format using self dicts
    def convert_back(self):
        graph = []
        for number in range(len(self._index_list) - 1):
            vertex = self._map_number_to_vertex[number]
            index = self._index_list[number]
            while index < self._index_list[number + 1]:
                to_vertex = self._map_number_to_vertex[self._neighbors_list[index]]
                weight = self._weights_list[index]
                edge = [vertex, to_vertex, weight]
                graph.append(edge)
                index += 10.
        return graph

    # sort the neighbors for each vertex
    def sort_neighbors(self):
        for number in range(len(self._index_list) - 1):
            start = self._index_list[number]
            end = self._index_list[number + 1]
            if self.weighted:
                if start == 230:
                    print()
                neighbors_weights = {self._neighbors_list[i]: self._weights_list[i] for i in range(start, end)}
                neighbors_weights = OrderedDict(sorted(neighbors_weights.items()))
                self._neighbors_list[start: end] = neighbors_weights.keys()
                self._weights_list[start: end] = neighbors_weights.values()
            else:
                self._neighbors_list[start: end] = sorted(self._neighbors_list[start: end])

    # get neighbors of specific node n
    def neighbors(self, vertex):
        node = self._map_vertex_to_number[vertex]
        idx = self._index_list[node]
        neighbors_list = []
        weights_list = []
        while idx < self._index_list[node + 1]:
            neighbors_list.append(self._map_number_to_vertex[self._neighbors_list[idx]])
            if self.weighted:
                weights_list.append(self._weights_list[idx])
            idx += 1
        if self.weighted:
            return neighbors_list, weights_list
        else:
            return neighbors_list

    # get neighbors and weights for every vertex
    def graph_adjacency(self):
        graph_adjacency_dict = dict()
        for number in range(len(self._index_list) - 1):
            vertex = self._map_number_to_vertex[number]
            if vertex not in graph_adjacency_dict.keys():
                graph_adjacency_dict[vertex] = dict()

            if self.weighted:
                neighbors_list, weights_list = self.neighbors(vertex)
                for neighbor, weight in zip(neighbors_list, weights_list):
                    graph_adjacency_dict[vertex][neighbor] = {'weight': weight}
            else:
                graph_adjacency_dict[vertex] = self.neighbors(vertex)
        return graph_adjacency_dict

if __name__ == '__main__':
    # graph = np.array([[0, 1, 1], [0, 2, 2], [0, 3, 3], [3, 2, 32], [2, 0, 20], [3, 1, 31]])  # code should work on this
    # l1, l2, l3, t1, t2 = weighted_directional_convert(graph)

    # graph = np.array([[5, 1, 51], [2, 3, 23], [5, 3, 53], [4, 5, 45]])
    # l1, l2, l3, t1, t2 = weighted_directional_convert(graph)
    # rootDir = os.path.join(os.path.join("", "PathwayProbabilitiesCalculation", "data", "yoram_network_1"))
    # graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
    #                    os.walk(rootDir) for file in filenames]
    #
    list_of_list_graph = lol_graph()
    graph_filenames = ["C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_4\\yoram_network_4_graph_1_01.csv"]
    #                    # "C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_1\\yoram_network_1_graph_1_10.csv",
    #                    # "C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_1\\yoram_network_1_graph_2_01.csv",
    #                    # "C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_1\\yoram_network_1_graph_2_10.csv",
    #                    # "C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_1\\yoram_network_1_graph_3_01.csv",
    #                    # "C:\\Users\\User\\PycharmProjects\\git_code\\MultipartiteCommunityDetection\\data\\yoram_network_1\\yoram_network_1_graph_3_10.csv"]
    #
    list_of_list_graph.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)], directed=True, weighted=True)
    # list_of_list_graph.convert([[1,2,90], [3,2, 190], [3, 1, 290], [2, 1, 390]], directed=True, weighted=True)

    # print(list_of_list_graph._index_list)
    # print(list_of_list_graph._index_list)
    # print(list_of_list_graph._index_list)
    # print("index list", list_of_list_graph._index_list)
    # print("neighbors list", list_of_list_graph._neighbors_list)
    # print("weights list", list_of_list_graph._weights_list)
    # print("map_vertex_to_number", list_of_list_graph._map_vertex_to_number)
    # print("map_number_to_vertex", list_of_list_graph._map_number_to_vertex)
    # print("BACK "+ str(list_of_list_graph.convert_back()))
    print("graph.adj - ", list_of_list_graph.graph_adjacency())
    # print(list_of_list_graph.graph_adjacency())
    # ns = convert_back(graph)
    # print(ns)
    # print(neighbors(5, l1, l2, t1, t2))
