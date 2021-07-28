import csv
import os
import time
from random import sample

import networkx as nx

from BipartiteProbabilisticMatching.matching_solutions import MatchingProblem

import numpy as np

from PathwayProbabilitiesCalculation.pathway_probabilities_calculation import iterate_by_layers, normalize_probs_matrix, \
    top5_probs_to_csv
from PathwayProbabilitiesCalculation.probabilities_using_embeddings import fun_3, node2vec_embed, createKD, \
    get_closest_neighbors
from multipartite_lol_graph import MultipartiteLol
from StaticGraphEmbeddings.evaluation_tasks.calculate_static_embeddings import ogre_static_embeddings


SEP = '/'

class Task:
    def __init__(self, results_root='.', task_params=None):
        self.results_root = results_root
        self.task_params = task_params
        self.destination = None
        self.results_dir = None
        self.results_files = None
        self.runtime_file = None
        self.memory_file = None
        self.eval_files = None
        self.data_name = None
        self.string_list = None
        self.eval_methods = None
        self.scores = []
        self.runtimes = []
        self.ids = []
        self.ids = []

        if not self.task_params:
            self.task_params = {}

    def prepare(self, graph_params, eval=False):
        self.data_name = graph_params.data_name
        self.destination = SEP.join([self.results_root, "task_" + str(self), self.data_name])
        self.string_list = [self.task_params.get('embedding')]
        self.string_list = [string for string in self.string_list if string]
        self.runtime_file = SEP.join([self.destination, "_".join(self.string_list + ["runtime.csv"])])
        self.memory_file = SEP.join([self.destination, "_".join(self.string_list + ["memory.csv"])])
        results_destination = SEP.join([self.destination, "_".join(self.string_list + ["results"])])
        self.results_dir = SEP.join([results_destination, "_".join([graph_params.name, "results"])])
        if not os.path.exists(self.results_dir) and not eval:
            os.makedirs(self.results_dir)
        elif not os.path.exists(self.results_dir):
            raise Exception("results not found")
        self.results_files = self.get_results_files(graph_params, eval=eval)
        self.eval_files = self.get_eval_files()
        self.ids.append(graph_params.id)

    def get_results_files(self, graph_params, eval=False):
        if eval:
            return [SEP.join([self.results_dir, file_name]) for file_name in os.listdir(self.results_dir) if
                    'gt.csv' not in file_name]
        else:
            return [SEP.join([self.results_dir, file_name]) for file_name in os.listdir(graph_params.path) if
                    'gt.csv' not in file_name]

    def get_eval_files(self):
        return {method: SEP.join([self.destination, "_".join(self.string_list + [method]) + ".csv"])
                for method in self.eval_methods}

    def save_attributes(self, memory):
        self.save_to_file([['x'] + self.ids, [self.data_name] + self.runtimes], self.runtime_file)
        self.save_to_file([['x'] + self.ids, [self.data_name] + memory], self.memory_file)

    def save_eval(self, method):
        self.save_to_file([['x'] + self.ids, [self.data_name] + self.scores], self.eval_files[method])

    def clean(self):
        self.scores = []
        self.runtimes = []
        self.ids = []

    def save_to_file(self, lines_list, path):
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            for line in lines_list:
                wr.writerow(line)


class BipartiteProbabilisticMatchingTask(Task):
    def __init__(self, results_root='.', task_params=None):
        super().__init__(results_root, task_params)
        self.eval_methods = ['avg_acc', 'winner_acc', 'top5_acc']

    def run(self, graph_params):
        start = time.time()
        print("Running task",str(self), 'on graph', graph_params.name)
        self.prepare(graph_params)

        # file_names = sys.argv[2:len(sys.argv)]
        first_stage_params = {"rho_0": self.task_params.get("rho_0", 0.3),
                              "rho_1": self.task_params.get("rho_1", 0.6),
                              "epsilon": self.task_params.get("epsilon", 1e-2)}
        # first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
        #                                          "yoram_network_1",
        #                                          f"yoram_network_1_graph_{g}.csv") for g in range(1, 4)]
        for graph_path, first_saving_path in zip(graph_params.files, self.results_files):
            first_saving_path_01 = first_saving_path[:-4] + "_01" + first_saving_path[-4:]
            first_saving_path_10 = first_saving_path[:-4] + "_10" + first_saving_path[-4:]
            MatchingProblem(graph_path, "flow_numeric", first_stage_params, first_saving_path_01, row_ind=0, col_ind=1)
            MatchingProblem(graph_path, "flow_numeric", first_stage_params, first_saving_path_10, row_ind=1, col_ind=0)

        # plot_toy_graphs(file_names=file_names, name="small", graphs_directions=[(0, 1)], problem=[4, 16])
        # plot_toy_graphs(file_names=[first_saving_path_01], name="small_01", directed=True, graphs_directions=[(0, 1)], header=True, integer=False, problem=[0.18, 0.79])
        # plot_toy_graphs(file_names=[first_saving_path_10], name="small_10", directed=True, graphs_directions=[(1, 0)], header=True, integer=False, problem=[0.84, 0.17])

        self.runtimes.append(time.time() - start)

    def eval(self, graph_params, method):
        # print("Evaluating task", str(self), 'on graph', graph_params.name, 'with method', method)
        if method not in self.eval_methods:
            raise Exception('method', method, 'not found!')
        self.prepare(graph_params, eval=True)
        if not graph_params.gt:
            files_scores = []
            for file in self.results_files:
                with open(file, "r", newline='') as csvfile:
                    probs = {}
                    datareader = csv.reader(csvfile)
                    next(datareader, None)  # skip the headers
                    for edge in datareader:
                        probs[edge[0]] = probs.get(edge[0], {})
                        probs[edge[0]][edge[1]] = float(edge[2])
                    for key, value in probs.items():
                        probs[key] = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))

                    if method == 'avg_acc':
                        scores = [neighbors.get(node, 0) for node, neighbors in probs.items()]
                    elif method == 'winner_acc':
                        scores = [1 if node == list(neighbors.keys())[0] else 0 for node, neighbors in probs.items()]
                    elif method == 'top5_acc':
                        scores = [1 if node in list(neighbors.keys())[:5] else 0 for node, neighbors in probs.items()]
                    else:
                        raise Exception('method'+ method+ 'not found!')
                    files_scores.append(np.mean(scores))
            score = 100 * np.mean(files_scores)
        else:
            with open(graph_params.gt, "r", newline='') as csvfile:
                gt = list(csv.reader(csvfile))
                print()
            score = 0
        self.scores.append(score)


    def __str__(self):
        return '1_BipartiteProbabilisticMatching'


class MultipartiteCommunityDetectionTask(Task):
    def __init__(self, results_root='.', task_params=None):
        super().__init__(results_root, task_params)
        self.eval_methods = ['full_avg_acc', 'all_avg_acc']

    def run(self, graph_params):
        start = time.time()
        print("Running task",str(self), 'on graph', graph_params.name)
        self.prepare(graph_params)

        lol = True
        if lol:
            from MultipartiteCommunityDetection.run_louvain_lol import run_louvain
        else:
            from MultipartiteCommunityDetection.run_louvain import run_louvain, load_graph_from_files
        np.random.seed(42)

        if lol:
            graph = MultipartiteLol()
            graph.convert_with_csv(graph_params.files, graph_params.from_to_ids)
            graph.set_nodes_type_dict()
        else:
            graph = load_graph_from_files(graph_params.files, graph_params.from_to_ids, has_title=True, cutoff=0.0)

        params = {"graph": graph,
                  "dump_name": self.results_files[0],
                  "res": self.task_params.get("res", 0.),
                  "beta": self.task_params.get("beta", [10., 10., 10.]),
                  "assess": self.task_params.get("assess", False),
                  "ground_truth": self.task_params.get("ground_truth", None),
                  "draw": self.task_params.get("draw", False)}

        run_louvain(**params)

        self.runtimes.append(time.time() - start)

    def eval(self, graph_params, method):
        # print("Evaluating task", str(self), 'on graph', graph_params.name, 'with method', method)
        if method not in self.eval_methods:
            raise Exception('method', method, 'not found!')
        self.prepare(graph_params, eval=True)
        results_file = self.results_files[0]
        num_of_groups = self.task_params.get('num_of_groups', 3)
        if not graph_params.gt:
            with open(results_file, "r", newline='') as csvfile:
                coms = {}
                datareader = csv.reader(csvfile)
                next(datareader, None)  # skip the headers
                for row in datareader:
                    if len(row) != 3:
                        continue
                    coms[row[2]] = coms.get(row[2], [])
                    coms[row[2]].append(row[1] + '_' + row[0])

                scores = []
                # avg between all the groups that have a full match (for example only [0_22, 1_22, 2_22])
                if method == 'full_avg_acc':
                    for c, nodes in coms.items():
                        nodes_com, nodes_idx = zip(*[node.split("_") for node in nodes])
                        # check if nodes_idx has only one unique value, it means the community is good
                        if len(nodes) == num_of_groups and len(set(nodes_idx)) == 1:
                            scores.append(1)
                        elif len(nodes) == num_of_groups:
                            scores.append(0)
                # avg between all the groups (for example [0_22, 1_22, 2_22], [0_22, 2_22])
                elif method == 'all_avg_acc':
                    for c, nodes in coms.items():
                        nodes_com, nodes_idx = zip(*[node.split("_") for node in nodes])
                        # check if nodes_idx has only one unique value, it means the community is good
                        if len(nodes) == num_of_groups and len(set(nodes_idx)) == 1:
                            scores.append(1)
                        else:
                            scores.append(0)
                else:
                    raise Exception('method', method, 'not found!')
            score = 100 * np.mean(scores)
        else:
            with open(graph_params.gt, "r", newline='') as csvfile:
                gt = list(csv.reader(csvfile))
                print()
            score = 0
        self.scores.append(score)

    def get_results_files(self, graph_params, eval=False):
        if not eval:
            return [SEP.join([self.results_dir, os.path.basename(self.results_dir)]) + '.csv']
        else:
            return [SEP.join([self.results_dir, file_name]) for file_name in os.listdir(self.results_dir) if
                    'gt.csv' not in file_name]


    def __str__(self):
        return '2_MultipartiteCommunityDetection'


class PathwayProbabilitiesCalculationTask(Task):
    def __init__(self, results_root='.', task_params=None):
        super().__init__(results_root, task_params)
        self.eval_methods = ['avg_acc', 'norm_avg_acc', 'winner_acc', 'top5_acc']

    def run(self, graph_params):
        start = time.time()
        print("Running task", str(self), 'on graph', graph_params.name)
        self.prepare(graph_params)

        results_file = self.results_files[0]
        open(results_file, 'w').close()
        list_of_list_graph = MultipartiteLol()
        list_of_list_graph.convert_with_csv(graph_params.files, graph_params.from_to_ids)
        list_of_list_graph.set_nodes_type_dict()
        nodes = list_of_list_graph.nodes()
        starting_points = self.task_params.get('starting_points', 5)
        limit_of_steps = self.task_params.get('limit_of_steps', 4)
        sampled_nodes = sample(nodes, starting_points)
        for start_point in sampled_nodes:
            probs = iterate_by_layers(list_of_list_graph, limit_of_steps, start_point)
            passway_probability = normalize_probs_matrix(probs)
            top5_probs_to_csv(passway_probability, results_file, start_point)



        self.runtimes.append(time.time() - start)

    def eval(self, graph_params, method):
        # print("Evaluating task", str(self), 'on graph', graph_params.name, 'with method', method)
        if method not in self.eval_methods:
            raise Exception('method', method, 'not found!')
        self.prepare(graph_params, eval=True)
        results_file = self.results_files[0]
        if not graph_params.gt:
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
                if method == 'avg_acc':
                    scores = [neighbors.get(node.split('_')[1], 0) for node, groups in probs.items() for neighbors in
                              groups.values()]
                elif method == 'norm_avg_acc':
                    scores = [
                        neighbors.get(node.split('_')[1], 0) / sum(neighbors.values()) if sum(neighbors.values()) else 0
                        for node, groups in probs.items() for neighbors in groups.values()]
                elif method == 'winner_acc':
                    scores = [1 if node.split('_')[1] == list(neighbors.keys())[0] else 0 for node, groups in
                              probs.items() for neighbors in groups.values()]
                elif method == 'top5_acc':
                    scores = [1 if node.split('_')[1] in list(neighbors.keys())[:5] else 0 for node, groups in
                              probs.items() for neighbors in groups.values()]
                else:
                    raise Exception('method', method, 'not found!')
            score = 100 * np.mean(scores)
        else:
            with open(graph_params.gt, "r", newline='') as csvfile:
                gt = list(csv.reader(csvfile))
                print()
            score = 0
        self.scores.append(score)

    def get_results_files(self, graph_params, eval=False):
        if not eval:
            return [SEP.join([self.results_dir, os.path.basename(self.results_dir)]) + '.csv']
        else:
            return [SEP.join([self.results_dir, file_name]) for file_name in os.listdir(self.results_dir) if
                    'gt.csv' not in file_name]

    def __str__(self):
        return '3_PathwayProbabilitiesCalculation'

class ProbabilitiesUsingEmbeddingsTask(Task):
    def __init__(self, results_root='.', task_params=None):
        super().__init__(results_root, task_params)
        self.eval_methods = ['avg_acc', 'norm_avg_acc', 'winner_acc', 'top5_acc']

    def run(self, graph_params):
        start = time.time()
        print("Running task", str(self), 'on graph', graph_params.name)
        self.prepare(graph_params)

        results_file = self.results_files[0]
        embedding = self.task_params.get('embedding', 'node2vec')
        epsilon = self.task_params.get('epsilon', 0.1)
        num_of_groups = self.task_params.get('num_of_groups', 3)

        open(results_file, 'w').close()
        distance_functions = [fun_3]  # [fun_1, fun_2, fun_3, fun_4]
        acc_dict = {fun: [] for fun in distance_functions}
        index_count = [0] * 6
        # print(directory)
        if embedding == 'ogre':
            z, graph, initial_size, list_initial_proj_nodes = ogre_static_embeddings(graph_params.files, graph_params.from_to_ids, epsilon)
            node_to_embed = z['OGRE + node2vec'].list_dicts_embedding[0]
            # node_to_embed = z['node2vec'][1]
        elif embedding == 'node2vec':
            try:
                node_to_embed, graph = node2vec_embed(graph_params.files, graph_params.from_to_ids)
            except:
                raise Exception("node2vec crashed...")
        else:
            raise Exception("embedding " + str(embedding) + " was not found")
        tree, dic = createKD(node_to_embed)
        for idx, node in dic.items():
            identity = node.split("_")[1]
            k = 11
            closest_neighbors, counts = get_closest_neighbors(tree, node, k, node_to_embed, dic, num_of_groups)
            while min(counts.values()) < 5:
                k += 10
                closest_neighbors, counts = get_closest_neighbors(tree, node, k, node_to_embed, dic, num_of_groups)
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
                    if nodes_sum != 0:
                        distances_after_norm = [i / nodes_sum for i in list(nodes_after_fun.values())]
                    else:
                        distances_after_norm = [1 if i == min(list(nodes.items())[:5]) else 0 for i in
                                                list(nodes.items())[:5]]
                    nodes_after_norm = {node: prob for node, prob in zip(nodes_after_fun.keys(), distances_after_norm)}
                    # prob = nodes_after_norm.get(identity, 0)
                    # acc_dict[fun].append(prob)
                probs[group] = nodes_after_norm
                # print("function 3 got acc", prob)
            top5_probs_to_csv(probs, results_file, node)


        self.runtimes.append(time.time() - start)

    def eval(self, graph_params, method):
        if method not in self.eval_methods:
            raise Exception('method', method, 'not found!')
        self.prepare(graph_params, eval=True)
        results_file = self.results_files[0]

        if not graph_params.gt:
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
                if method == 'avg_acc':
                    scores = [neighbors.get(node.split('_')[1], 0) for node, groups in probs.items() for neighbors in
                              groups.values()]
                elif method == 'norm_avg_acc':
                    scores = [neighbors.get(node.split('_')[1], 0) / sum(neighbors.values()) for node, groups in
                              probs.items()
                              for neighbors in groups.values()]
                elif method == 'winner_acc':
                    scores = [1 if node.split('_')[1] == list(neighbors.keys())[0] else 0 for node, groups in
                              probs.items() for
                              neighbors in groups.values()]
                elif method == 'top5_acc':
                    scores = [1 if node.split('_')[1] in list(neighbors.keys())[:5] else 0 for node, groups in
                              probs.items() for
                              neighbors in groups.values()]
                else:
                    raise Exception('method', method, 'not found!')
            score = 100 * np.mean(scores)

        else:
            with open(graph_params.gt, "r", newline='') as csvfile:
                gt = list(csv.reader(csvfile))
                print()
            score = 0
        self.scores.append(score)

    def get_results_files(self, graph_params, eval=False):
        if not eval:
            return [SEP.join([self.results_dir, os.path.basename(self.results_dir)]) + '.csv']
        else:
            return [SEP.join([self.results_dir, file_name]) for file_name in os.listdir(self.results_dir) if
                    'gt.csv' not in file_name]

    def __str__(self):
        return '4_ProbabilitiesUsingEmbeddings'