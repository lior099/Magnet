import os
import numpy as np
from collections import OrderedDict
import csv


class lol_graph:

    def __init__(self, index_list=[0], neighbors_list=[], weights_list=[], map_vertex_to_number=OrderedDict(),
                 map_number_to_vertex=OrderedDict(), directed=False, weighted=False):
        self._index_list = index_list
        self._neighbors_list = neighbors_list
        self._weights_list = weights_list
        self._map_vertex_to_number = map_vertex_to_number
        self._map_number_to_vertex = map_number_to_vertex
        self.directed = directed
        self.weighted = weighted

    def number_of_edges(self):
        return len(self._neighbors_list)
        # return sum(1 for neighbor in self._neighbors_list if neighbor is not None)

    def number_of_vertices(self):
        return len(self._index_list) - 1
        # return sum(1 for vertex in self._index_list if vertex is not None) - 1

    def deep_copy(self):
        return lol_graph(self._index_list.copy(), self._neighbors_list.copy(), self._weights_list.copy(),
                         self._map_vertex_to_number.copy(), self._map_number_to_vertex.copy(), self.directed, self.weighted)
    # outdegree
    def sum_weight_from_node(self, node):
        if self.weighted:
            neighbors_list, weights_list = self.neighbors(node)
            return sum(weights_list)
        else:
            neighbors_list = self.neighbors(node)
            return len(neighbors_list)

    # Iterative Binary Search Function
    # It returns index of x in given array arr if present,
    # else returns -1
    def binary_search(self, arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2

            # Check if x is present at mid
            if arr[mid] < x:
                low = mid + 1

            # If x is greater, ignore left half
            elif arr[mid] > x:
                high = mid - 1

            # If x is smaller, ignore right half
            else:
                return mid

        # If we reach here, then the element was not present
        return -1

    # indegree
    def sum_weight_to_node(self, node):
        sum = 0
        for vertex in self.get_vertices_list():
            if self.weighted:
                neighbors_list, weights_list = self.neighbors(vertex)
            else:
                neighbors_list = self.neighbors(vertex)
            x = self.binary_search(neighbors_list, node)
            if x is not -1:
                if self.weighted:
                    sum += weights_list[x]
                else:
                    sum += 1
        return sum

    def all_nodes_directed_to_node(self, node):
        nodes_list = []
        for vertex in self.get_vertices_list():
            neighbors_list, weights_list = self.neighbors(vertex)
            x = self.binary_search(neighbors_list, node)
            if x is not -1:
                nodes_list.append(vertex)
        return nodes_list

    def get_vertices_list(self):
        return self._map_vertex_to_number.keys()

    def get_edge_list(self):
        return self.convert_back()

    def is_edge_between_nodes(self, node1, node2):
        neighbors_list, weights_list = self.neighbors(node1)
        x = self.binary_search(neighbors_list, node2)
        return x is not -1

    def get_weight_of_edge(self, node1, node2):
        if self.weighted:
            neighbors_list, weights_list = self.neighbors(node1)
            x = self.binary_search(neighbors_list, node2)
            if x is not -1:
                return weights_list[x]
            print("There is no edge between ", node1, " and ", node2)
        else:
            print("Notice: The graph is not weighted")

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
            if not self.directed:
                d[self._map_vertex_to_number[edge[1]]] = d.get(self._map_vertex_to_number[edge[1]], 0) + 1
            elif self._map_vertex_to_number[edge[1]] not in d.keys():
                d[self._map_vertex_to_number[edge[1]]] = 0

        '''transfer the dictionary to list'''
        for j in range(1, len(d.keys()) + 1):
            self._index_list.append(self._index_list[j - 1] + d.get(j - 1, 0))

        '''create the second list'''
        if self.directed:
            self._neighbors_list = [-1] * len(graph)
        else:
            self._neighbors_list = [-1] * len(graph) * 2
        if self.weighted:
            self._weights_list = [0] * len(graph)

        space = OrderedDict((x, -1) for x in self._map_number_to_vertex.keys())
        for idx, edge in enumerate(graph):
            left = self._map_vertex_to_number[edge[0]]
            right = self._map_vertex_to_number[edge[1]]
            if self.weighted:
                weight = float(edge[2])

            if space[left] != -1:
                space[left] += 1
                i = space[left]
            else:
                i = self._index_list[left]
                space[left] = i
            self._neighbors_list[i] = right
            if self.weighted:
                self._weights_list[i] = weight

            if not self.directed:
                if space[right] != -1:
                    space[right] += 1
                    i = space[right]
                else:
                    i = self._index_list[right]
                    space[right] = i
                self._neighbors_list[i] = left
                if self.weighted:
                    self._weights_list[i] = weight
        self._neighbors_list, self._weights_list = self.sort_all(index_list=self._index_list,
                                                                 neighbors_list=self._neighbors_list,
                                                                 weights_list=self._weights_list)

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
                index += 1
        return graph

    # sort the neighbors for each vertex
    def sort_all(self, index_list=None, neighbors_list=None, weights_list=None):
        for number in range(len(index_list) - 1):
            start = index_list[number]
            end = index_list[number + 1]
            neighbors_list[start: end], weights_list[start: end] = self.sort_neighbors(neighbors_list[start: end], weights_list[start: end])
        return neighbors_list, weights_list

    def sort_neighbors(self, neighbors_list=None, weights_list=None):
        if self.weighted:
            neighbors_weights = {neighbors_list[i]: weights_list[i] for i in range(len(neighbors_list))}
            neighbors_weights = OrderedDict(sorted(neighbors_weights.items()))
            return neighbors_weights.keys(), neighbors_weights.values()
        else:
            return sorted(neighbors_list), weights_list


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

    # Add new edges to the graph, but with limitations:
    # For example, if the edge is [w,v] and the graph is diracted, w can't be an existing vertex.
    # if the graph is not diracted, w and v can't be existing vertices.
    def add_edges(self, edges):
        last_index = self._index_list[-1]
        index_list = [last_index]
        neighbors_list = []
        weights_list = []

        vertices_amount = len(self._map_vertex_to_number.keys())
        free = vertices_amount
        map_vertex_to_number = OrderedDict()

        '''create dictionary to self._map_vertex_to_number from edges to our new numbering'''
        for edge in edges:
            if (self._map_vertex_to_number.get(edge[0], None) is not None or
                (not self.directed and self._map_vertex_to_number.get(edge[1], None) is not None)):
                print("Error: add_edges can't add edges from an existing vertex")
                return
            if map_vertex_to_number.get(edge[0], None) is None:
                map_vertex_to_number[edge[0]] = free
                free += 1
            if map_vertex_to_number.get(edge[1], None) is None and self._map_vertex_to_number.get(edge[1], None) is None:
                map_vertex_to_number[edge[1]] = free
                free += 1
        '''create the opposite dictionary'''
        map_number_to_vertex = OrderedDict((y, x) for x, y in map_vertex_to_number.items())

        """update the original dicts"""
        self._map_vertex_to_number.update(map_vertex_to_number)
        self._map_number_to_vertex.update(map_number_to_vertex)

        d = OrderedDict()
        '''starting to create the index list. Unordered is important'''
        for idx, edge in enumerate(edges):
            d[self._map_vertex_to_number[edge[0]]] = d.get(self._map_vertex_to_number[edge[0]], 0) + 1
            if not self.directed:
                d[self._map_vertex_to_number[edge[1]]] = d.get(self._map_vertex_to_number[edge[1]], 0) + 1
            elif self._map_vertex_to_number[edge[1]] not in d.keys() and edge[1] in map_vertex_to_number.keys():
                d[self._map_vertex_to_number[edge[1]]] = 0

        '''transfer the dictionary to list'''
        for j in range(1, len(d.keys()) + 1):
            index_list.append(index_list[j - 1] + d.get(vertices_amount + j - 1, 0))

        '''create the second list'''
        if self.directed:
            neighbors_list = [-1] * len(edges)
        else:
            neighbors_list = [-1] * len(edges) * 2
        if self.weighted:
            weights_list = [0] * len(edges)

        space = OrderedDict((x, -1) for x in self._map_number_to_vertex.keys())
        for idx, edge in enumerate(edges):
            left = self._map_vertex_to_number[edge[0]]
            right = self._map_vertex_to_number[edge[1]]
            if self.weighted:
                weight = float(edge[2])

            if space[left] != -1:
                space[left] += 1
                i = space[left]
            else:
                i = index_list[left - vertices_amount] - last_index
                space[left] = i
            neighbors_list[i] = right
            if self.weighted:
                weights_list[i] = weight

            if not self.directed:
                if space[right] != -1:
                    space[right] += 1
                    i = space[right]
                else:
                    i = index_list[right - len(self._index_list) - 1]
                    space[right] = i
                neighbors_list[i] = left
                if self.weighted:
                    weights_list[i] = weight

        """sort the neighbors"""
        neighbors_list, weights_list = self.sort_all(index_list=index_list,
                                                     neighbors_list=neighbors_list,
                                                     weights_list=weights_list)

        """update the original dicts"""
        self._index_list += index_list[1:]
        self._neighbors_list += neighbors_list
        self._weights_list += weights_list


    # swap between two edges, but with limitations.
    # For example, the graph can only be directed.
    # The swap can only be in the form of: from edge [a,b] to edge [a, c].
    def swap_edge(self, edge_to_delete, edge_to_add):
        if not self.directed or (self.directed and edge_to_delete[0] != edge_to_add[0]):
            print("Error: swap_edge can only be only on directed graph and from the same vertex")
            return
        if not self.is_edge_between_nodes(edge_to_delete[0], edge_to_delete[1]):
            print("Error: edge_to_delete was not found")
            return
        number = self._map_vertex_to_number[edge_to_add[0]]
        to_number = self._map_vertex_to_number[edge_to_add[1]]
        from_number = self._map_vertex_to_number[edge_to_delete[1]]
        start_index_of_source = self._index_list[number]
        end_index_of_source = self._index_list[number+1]

        neighbors_list = self._neighbors_list[start_index_of_source: end_index_of_source]
        neighbor_index = self.binary_search(neighbors_list, from_number)
        neighbors_list[neighbor_index] = to_number

        if self.weighted:
            weights_list = self._weights_list[start_index_of_source: end_index_of_source]
            weights_list[neighbor_index] = edge_to_add[2]
            neighbor_index, weights_list = self.sort_neighbors(neighbors_list, weights_list)
            self._weights_list[start_index_of_source: end_index_of_source] = weights_list
            # index_of_replacement = start_index_of_source + neighbor_index
            # neighbors_list[neighbor_index] = to_number
            # weights_list[neighbor_index] = edge_to_add[2]

        else:
            neighbor_index, weights_list = self.sort_neighbors(neighbors_list)
        self._neighbors_list[start_index_of_source: end_index_of_source] = neighbor_index




if __name__ == '__main__':
    list_of_list_graph = lol_graph()
    rootDir = os.path.join("..", "PathwayProbabilitiesCalculation", "data", "toy_network_1")
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                              os.walk(rootDir) for file in filenames]
    graph_filenames.sort()
    list_of_list_graph.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], True, True)
    # list_of_list_graph.convert([[5, 1, 51], [2, 3, 23], [5, 3, 53], [4, 5, 45]],  True, True)
    # print("index list", list_of_list_graph._index_list)
    # print("neighbors list", list_of_list_graph._neighbors_list)
    # print("weights list", list_of_list_graph._weights_list)
    # print("map_vertex_to_number", list_of_list_graph._map_vertex_to_number)
    # print("map_number_to_vertex", list_of_list_graph._map_number_to_vertex)
    # print("BACK "+ str(list_of_list_graph.convert_back()))
    # # print("graph.adj - ", list_of_list_graph.graph_adjacency())
    # print("number of edges", list_of_list_graph.number_of_edges())
    # print("number of vertices", list_of_list_graph.number_of_vertices())
    # print("sum_weight_from_node", list_of_list_graph.sum_weight_from_node("5a"))
    # print("sum_weight_to_node", list_of_list_graph.sum_weight_to_node("5a"))
    # print("all nodes to node", "8c", list_of_list_graph.all_nodes_directed_to_node("8c"))
    # print("edge list:", list_of_list_graph.get_edge_list())
    # print(list_of_list_graph.get_weight_of_edge("1a", "1b"))
