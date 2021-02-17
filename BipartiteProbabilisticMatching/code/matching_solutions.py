import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from memory_profiler import memory_usage, profile
import time
import csv
import sys



class MatchingProblem:
    algorithm_names = {
        "deg_update": "Updating by degree",
        "flow_analytic": "Flow - analytic solution",
        "flow_numeric": "Flow - numeric solution",
        "null_model": "Null model"
    }

    def __init__(self, graph_path, algorithm, params, path, row_ind=0, col_ind=1, matches_path=None):
        """
        The main class. Receives a graph and the ground truth values and implements the requested algorithm.
        The results can later be measured using top-k accuracy and the sum of probabilities score, or can be visualized.

        :param graph_path: The path to a csv file of the weighted biadjacency matrices.
        :param algorithm: The name of the desired algorithm. Can be one of the following four:
               "deg_update", "flow_analytic", "flow_numeric" or "null_model".
        :param params: The dictionary of the parameters required for the algorithm.
        :param path: The path in which the results are saved.
        :param row_ind: index of column of sources side in csv file.
        :param col_ind: index of column of targets side in csv file.
        :param matches_path: The path to a csv file of the ground truth matches.
        """
        if matches_path is not None:
            self.true_matches = self._load_matches(matches_path)
        self.algorithm_name = self.algorithm_names[algorithm]
        self._char_to_num_row, self._num_to_char_row, self._char_to_num_col, self._num_to_char_col = {}, {}, {}, {}
        self.is_mapped = False
        self.graph, self.shape = self.load_graph(graph_path, row_ind, col_ind)
        self.p_mat = self.algorithm(algorithm, params, is_normalized=True)
        self.save_res(path)

    def load_graph(self, file, row_ind=0, col_ind=1):
        """
        Load the bipartite graph and the shape of its biadjacency matrix
        :param file: The path to a csv file of the weighted biadjacency matrices.
        :param row_ind: index of rows in csv file.
        :param col_ind: index of columns in csv file.
        :return: A networkx Graph and a numpy array of the bipartite weight matrices.
        """
        data = pd.read_csv(file, header=None)
        rows = data[row_ind]
        columns = data[col_ind]

        if data.shape[1] <= 2:
            w = np.ones(shape=len(data[0]))
        else:
            w = data[2].values.astype(float)

        try:
            rows = rows.values.astype(int)
            columns = columns.values.astype(int)
            M = csr_matrix((w, (rows, columns)))
        except Exception:
            self.is_mapped = True
            self._char_to_num_row, self._num_to_char_row = self.create_mapping(data, 0)
            self._char_to_num_col, self._num_to_char_col = self.create_mapping(data, 1)
            rows = rows.apply(lambda x: self._char_to_num_row[x]).values
            columns = columns.apply(lambda x: self._char_to_num_col[x]).values
            M = csr_matrix((w, (rows, columns)))
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(M)
        # instead of size maybe return biadj_matrix?
        return graph, M.shape

    @staticmethod
    def create_mapping(df, col):
        chars = df[col].unique().tolist()
        chars = sorted(chars)
        char_to_num = {}
        num_to_char = {}
        for i, c in enumerate(chars):
            char_to_num[c] = i
            num_to_char[i] = c
        return char_to_num, num_to_char

    def algorithm(self, algorithm, params, is_normalized=True):
        """
        Import and implement the requested algorithm, to create a probability matrices of the same dimensions as the
        biadjacency matrices, where the (i, j)-th element represents the probability that the vertex i of the first side
        matches the vertex j of the second side.
        Note that p is normalized row-wise (i.e. sum_j (p_ij) = 1), but not necessarily column-wise.

        :param algorithm: The string indicating which algorithm to run. Can be one of the following four:
               "deg_update", "flow_analytic", "flow_numeric" or "null_model".
        :param params: The dictionary of the parameters required for the algorithm.
        :return: The final probability matrices p
        """
        if algorithm == "flow_analytic":
            from BipartiteProbabilisticMatching.code.flow_analytic import algorithm
        elif algorithm == "flow_numeric":
            from BipartiteProbabilisticMatching.code.flow_numeric import algorithm
        p = algorithm(self, params, is_normalized=is_normalized)
        return p

    @staticmethod
    def _load_matches(matches_path):
        """
        Load the matches file, as a numpy array where the first column represents the vertices of one side and the
        second column represents the matching (ground truth) vertices from the other side.
        NOTE: The toy models we created had vertices with indices 1, 2, ..., number_of_vertices_per_side.
        Therefore, we subtract 1 from the loaded file. To use this function as is, the vertices should be indexed
        accordingly.

        :param matches_path: The path to a csv file of the ground truth matches.
        :return: A numpy array of the matches.
        """
        # NOTE: Yoram's toy models are with vertices 1 to 100. If another format is used, generalize the function.
        return pd.read_csv(matches_path).to_numpy()

    def nonzero_degree_vertices(self):
        """
        Calculate and return the vertices with degree > 0. We can calculate the probability to them only.
        """
        return [v for v in range(self.unw_adj.shape[0]) if np.sum(self.unw_adj[v, :]) > 0]

    def top_k_accuracy(self, k):
        """
        Calculate the top-k accuracy.
        Here, top-k accuracy means taking the number of vertices from the first side for which the ground truth vertex
        appears in the top k vertices from the second side by the probability matrices, over the total number of vertices
        from the first side.

        :param k: int. The smaller k is, the more difficult it is to reach high accuracies.
        :return: The accuracy.
        """
        correct = 0.
        tried = 0.
        for v in range(self.p_mat.shape[0]):
            true = self.true_matches[v, 1]
            preds = np.argsort(- self.p_mat[v, :])[:k]
            if true in preds:
                correct += 1
            tried += 1
        return correct / tried * 100

    def sum_prob_score(self):
        """
        The sum of probabilities score, i.e. sum_i (p_[i, t]),
        where t is the index of the ground truth vertex and p is the probability matrices

        :return:
        """
        return sum([self.p_mat[i, self.true_matches[i, 1]] for i in range(self.true_matches.shape[0])])

    def visualize_results(self, saving_path):
        """
        Create and save a figure of the bipartite graph, with colored edges:
        Green edges represent true positive edges, blue represent true negative, orange represents false positive and
        red represents false negative.

        :param saving_path: The path in which the figure will be saved.
        """
        left_side_vertices = self.true_matches[:, 0]
        pos = nx.bipartite_layout(self.graph, left_side_vertices)
        edge_colors = []
        counts = [0, 0, 0, 0]
        for e in self.graph.edges:
            source = e[0]
            target = e[1] - self.true_matches.shape[0]  # The full-graph index -> index in the graph's second side.
            pred = np.argmax(self.p_mat[source, :])
            if self.true_matches[source, 1] == target:
                if pred == target:
                    # TP
                    edge_colors.append('g')
                    counts[0] += 1
                else:
                    # FN
                    edge_colors.append('r')
                    counts[3] += 1
            else:
                if pred == target:
                    # FP
                    edge_colors.append('orange')
                    counts[2] += 1
                else:
                    # TN
                    edge_colors.append('b')
                    counts[1] += 1
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors)
        plt.title(self.algorithm_name)
        plt.text(0, -0.75, 'TP (green) - %d, TN (blue) - %d, FP (orange) - %d, FN (red) - %d' %
                 (counts[0], counts[1], counts[2], counts[3]), fontsize=12,
                 horizontalalignment='center', verticalalignment='center',
                 bbox={'facecolor': 'grey', 'alpha': 0.7})
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.savefig(saving_path)

    def save_res(self, path):
        if self.is_mapped:
            edges = [(self._num_to_char_row[i], self._num_to_char_col[j], self.p_mat[i, j]) for i, j in
                     zip(*self.p_mat.nonzero())]
        else:
            edges = [(i, j, self.p_mat[i, j]) for i, j in zip(*self.p_mat.nonzero())]
        df = pd.DataFrame(edges, columns=["Source", "Target", "Probability"])
        df.to_csv(path, index=False)


def running_time():
    sizes = [10, 100, 500, 800, 1000, 3000, 5000, 8000, 10000, 12000, 15000, 18000, 20000] + list(
        np.arange(30000, 110000, 10000))
    times = []
    for size in sizes:
        params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2, "size": size}
        t = time.time()
        MatchingProblem(os.path.join("simulation", "experiments", "Experiment_{}_0.7_0.1.csv".format(size)),
                        "flow_numeric", params, path=os.path.join("new_data", "matrices", "edge_list_str_4_new.csv"),
                        matches_path=os.path.join("simulation", "real_tags", "Real_Tags_Exp_{}.csv".format(size)))
        estimate_time = time.time() - t
        print(size, ':', estimate_time)
        times.append(estimate_time)
    plt.plot(sizes[1:], times[1:], "-ok", color='red')
    plt.xlabel("Size")
    plt.ylabel("Running Time [s]")
    plt.title("Running Time As Function Of Size")
    plt.savefig("Running_Time.png")


def for_memory(size):
    params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2, "size": size}
    MatchingProblem(os.path.join("..", "data", "Experiment_{}_0.7_0.1.csv".format(size)),
                    "flow_numeric", params, path=os.path.join("new_data", "matrices", "edge_list_str_4_new.csv"),
                    matches_path=os.path.join("simulation", "real_tags", "Real_Tags_Exp_{}.csv".format(size)))


def memory_analysis():
    sizes = [10, 100, 500, 800, 1000, 3000, 5000, 8000, 10000, 12000, 15000, 18000, 20000] + list(
        np.arange(30000, 110000, 10000))
    memory_usage_list = []
    for size in sizes:
        k = memory_usage((for_memory, (size,)))[-1]
        print(k)
        memory_usage_list.append(k)
    plt.plot(sizes, memory_usage_list, "-ok", color='red')
    plt.xlabel("Size")
    plt.ylabel("Memory Usage [MB]")
    plt.title("Memory Usage As Function Of Size")
    plt.savefig("Memory_Usage.png")


def task1(file_names, first_stage_saving_paths, first_stage_params):
    # file_names = sys.argv[2:len(sys.argv)]
    # first_stage_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
    # first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
    #                                          "yoram_network_1",
    #                                          f"yoram_network_1_graph_{g}.csv") for g in range(1, 4)]
    start = time.time()
    for graph_path, first_saving_path in zip(file_names, first_stage_saving_paths):
        first_saving_path_01 = first_saving_path[:-4] + "_01" + first_saving_path[-4:]
        first_saving_path_10 = first_saving_path[:-4] + "_10" + first_saving_path[-4:]
        MatchingProblem(graph_path, "flow_numeric", first_stage_params, first_saving_path_01, row_ind=0, col_ind=1)
        MatchingProblem(graph_path, "flow_numeric", first_stage_params, first_saving_path_10, row_ind=1, col_ind=0)
    end = time.time()
    plot_toy_graphs(file_names=file_names, name="small", graphs_directions=[(0, 1)], problem=[4, 16])
    plot_toy_graphs(file_names=[first_saving_path_01], name="small_01", directed=True, graphs_directions=[(0, 1)], header=True, integer=False, problem=[0.18, 0.79])
    plot_toy_graphs(file_names=[first_saving_path_10], name="small_10", directed=True, graphs_directions=[(1, 0)], header=True, integer=False, problem=[0.84, 0.17])
    print(-start + end, "s")




def plot_toy_graphs(graph=None, file_names=None, header=False, partition=None, name='', labels=False, directed=False, graphs_directions=None, integer=True, problem=[None, None]):
    if file_names:
        edges = []
        nodes = []
        for i in range(len(file_names)):
            file = file_names[i]
            with open(file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                if header:
                    next(datareader, None)  # skip the headers
                for edge in datareader:
                    named_edge = [str(graphs_directions[i][0]) + "_" + edge[0],
                                  str(graphs_directions[i][1]) + "_" + edge[1], float(edge[2])]
                    edges.append(named_edge)
                    if named_edge[0] not in nodes:
                        nodes.append(named_edge[0])
                    if named_edge[1] not in nodes:
                        nodes.append(named_edge[1])
    if graph:
        edges = graph.convert_back()
        nodes = graph.nodes()
    nodes.sort()
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    # nodes[0], nodes[1] = nodes[1], nodes[0]
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    np.random.seed(2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    colors = {'0': 'r', '1': 'orange', '2': 'g'}
    colors = {'0': 'w', '1': 'w', '2': 'w'}
    all_colors = {str(i): ['b', 'g', 'r', 'c', 'm', 'y'] for i in range(3)}
    # node_colors = [colors[node.split("_")[0]] for node in nodes]
    shapes = {'0': 'o', '1': 's', '2': '^'}
    # node_shapes = [shapes[node.split("_")[0]] for node in nodes]
    nodes_dict = {}
    # node.split("_")[0]: node.split("_")[1] for node in nodes
    for node in nodes:
        if node.split("_")[0] not in nodes_dict.keys():
            nodes_dict[node.split("_")[0]] = [node]
        else:
            nodes_dict[node.split("_")[0]].append(node)
    if partition:
        colors = {key: [all_colors[key][partition[node]] for node in nodes] for key, nodes in nodes_dict.items()}
    labels = {node: node.split("_")[1] for node in nodes}
    # labels = {i: {node: node.split("_")[1] for node in nodes_dict[i]} for i in ['0', '1', '2']}
    # for i in "so^>v<dph8":
    nodePos = nx.layout.bipartite_layout(graph, nodes_dict['0'])
    nodePos = {node: pos[1] for pos, node in zip(nodePos.items(), nodes[::-1])}
    plt.rcParams["figure.figsize"] = (4, 4)
    if integer:
        edges_labels = {(node1, node2): int(weight['weight']) for node1, node2, weight in graph.edges(data=True)}
    else:
        edges_labels = {(node1, node2): round(weight['weight'], 2) for node1, node2, weight in graph.edges(data=True)}
    edges_labels1 = {edge: weight for edge, weight in edges_labels.items() if weight not in problem}
    edges_labels2 = {edge: weight for edge, weight in edges_labels.items() if weight == problem[0]}
    edges_labels3 = {edge: weight for edge, weight in edges_labels.items() if weight == problem[1]}
    nx.draw(graph, nodePos, arrowsize=20)
    for i, node_list in nodes_dict.items():
        nx.draw_networkx_nodes(graph, nodePos, nodelist=node_list, node_shape=shapes[i], node_color=colors[i], node_size=500, edgecolors='black')
    nx.draw_networkx_edges(graph, nodePos)
    nx.draw_networkx_edge_labels(graph, nodePos, edge_labels=edges_labels1, label_pos=0.5)
    nx.draw_networkx_edge_labels(graph, nodePos, edge_labels=edges_labels2, label_pos=0.4)
    nx.draw_networkx_edge_labels(graph, nodePos, edge_labels=edges_labels3, label_pos=0.62)
    nx.draw_networkx_labels(graph, nodePos, labels, font_size=16)
    # plt.figure(figsize=(20, 20))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig("toy_graph_"+name)
    plt.show()


    print()




if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    res_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    parameters = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
    mp = MatchingProblem(os.path.join(data_path, "Obs_Pair_K_Network_1_Graph_1.csv"), "flow_numeric", parameters,
                         path=os.path.join(res_path, "edge_list_1_1.csv"))
    mp.visualize_results("visualization_example.png")
    # running_time()
