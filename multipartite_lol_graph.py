import os
import numpy as np
from collections import OrderedDict
from lol_graph import lol_graph
import csv


class Multipartite_Lol(lol_graph):
    def __init__(self, groups_number=0, index_list=[0], neighbors_list=[], weights_list=[], map_node_to_number=OrderedDict(),
                 map_number_to_node=OrderedDict(), directed=False, weighted=False):
        super().__init__(index_list=[0], neighbors_list=[], weights_list=[], map_node_to_number=OrderedDict(),
                         map_number_to_node=OrderedDict(), directed=False, weighted=False)
        self.groups_number = groups_number

    # input: csv file containing edges list, in the form of [[5,1],[2,3],[5,3],[4,5]]
    def convert_with_csv(self, files_name, graphs_directions, directed=False, weighted=False, header=True):
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


if __name__ == '__main__':
    list_of_list_graph = Multipartite_Lol()
    rootDir = os.path.join("..", "PathwayProbabilitiesCalculation", "data", "toy_network_1")
    graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                              os.walk(rootDir) for file in filenames]
    graph_filenames.sort()
    list_of_list_graph.convert_with_csv(graph_filenames, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], True, True)
    print(list_of_list_graph.groups_number)
