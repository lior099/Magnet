import os
import numpy as np
from collections import OrderedDict
from lol_graph import lol_graph
import csv


class Multipartite_Lol(lol_graph):
    def __init__(self, groups_number=0, directed=False, weighted=False):
        super().__init__(directed=directed, weighted=weighted)
        self.groups_number = groups_number
        self.nodes_type_dict = {}

    def convert_with_csv(self, files_name, graphs_directions=None, directed=False, weighted=False, header=True):
        s = set()
        for t in graphs_directions:
            s.add(t[0])
            s.add(t[1])
        self.groups_number = len(s)

        self._map_node_to_number = OrderedDict()
        graph = []
        for i in range(len(files_name)):
            file = files_name[i]
            with open(file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                if header:
                    next(datareader, None)  # skip the headers
                for edge in datareader:
                    named_edge = [str(graphs_directions[i][0]) + "_" + edge[0],
                                  str(graphs_directions[i][1]) + "_" + edge[1]]
                    if weighted:
                        named_edge.append(float(edge[2]))
                    graph.append(named_edge)
                csvfile.close()
        self.convert(graph, directed, weighted)

    def return_node_type(self, node):
        return self.nodes_type_dict[node]['type']

    def copy(self):
        new_mp_lol_graph = Multipartite_Lol()
        new_mp_lol_graph._index_list = self._index_list.copy()
        new_mp_lol_graph._neighbors_list = self._neighbors_list.copy()
        new_mp_lol_graph._weights_list = self._weights_list.copy()
        new_mp_lol_graph._map_node_to_number = self._map_node_to_number.copy()
        new_mp_lol_graph._map_number_to_node = self._map_number_to_node.copy()
        new_mp_lol_graph.directed = self.directed
        new_mp_lol_graph.weighted = self.weighted
        new_mp_lol_graph.groups_number = self.groups_number
        new_mp_lol_graph.nodes_type_dict = self.nodes_type_dict
        return new_mp_lol_graph

    def set_nodes_type_dict(self):
        for node in self.nodes():
            node_type = int(node[0])
            type = [1 if i == node_type else 0 for i in range(self.groups_number)]
            self.nodes_type_dict[node] = {'type': type}

    def initialize_nodes_type_dict(self):
        for node in self.nodes():
            self.nodes_type_dict[node] = {}


if __name__ == '__main__':
    list_of_list_graph = Multipartite_Lol()
    rootDir = os.path.join("PathwayProbabilitiesCalculation", "data", "toy_network_1")
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                       os.walk(rootDir) for file in filenames]
    graph_filenames.sort()
    list_of_list_graph.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], True, True)
    print(list_of_list_graph.nodes())
