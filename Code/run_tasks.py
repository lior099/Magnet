import os
import random
import time
import sys

from Code.graphs import Params
from Code.tasks import BipartiteProbabilisticMatchingTask, MultipartiteCommunityDetectionTask, \
    PathwayProbabilitiesCalculationTask, ProbabilitiesUsingEmbeddingsTask

sys.path.append(os.path.abspath('..'))
from PathwayProbabilitiesCalculation.pathway_probabilities_calculation import task3, eval_task3
from PathwayProbabilitiesCalculation.probabilities_using_embeddings import task4, eval_task4
import matplotlib.pyplot as plt
from BipartiteProbabilisticMatching.matching_solutions import task1, eval_task1

import numpy as np
from memory_profiler import memory_usage
import csv

from multipartite_lol_graph import MultipartiteLol

lol = True
if lol:
    from MultipartiteCommunityDetection.run_louvain_lol import task2, eval_task2
else:
    from MultipartiteCommunityDetection.run_louvain import task2, load_graph_from_files

SEP = '/'

def get_data_path(task, results_root, data_name):
    if '1' in str(task):
        data_path = SEP.join([results_root, 'data', data_name])
    else:
        data_path = SEP.join([results_root, 'task_1_BipartiteProbabilisticMatching', data_name, 'results'])
    if not os.path.exists(data_path):
        raise Exception("data not found")
    return data_path


def get_graphs_paths(data_path):
    graphs_paths = [SEP.join([data_path, graph_dir]) for graph_dir in os.listdir(data_path)]
    graphs_names = ['_'.join(dir_name.split('_')[:-1]) if "_results" in dir_name else dir_name for dir_name in
                    os.listdir(data_path)]
    graphs_ids = [float(name.split('_')[0].replace(',', '.')) for name in graphs_names]
    return graphs_paths, graphs_names, graphs_ids


def get_graphs_params(task, results_root, data_name):
    graphs_params = []
    data_path = get_data_path(task, results_root, data_name)
    graphs_paths, graphs_names, graphs_ids = get_graphs_paths(data_path)
    for graph_path, graph_name, graph_id in zip(graphs_paths, graphs_names, graphs_ids):
        graph_files = [SEP.join([graph_path, file_name]) for file_name in os.listdir(graph_path) if 'gt.csv' not in file_name]
        gt_file = [SEP.join([graph_path, file_name]) for file_name in os.listdir(graph_path) if 'gt.csv' in file_name]
        num_of_groups = task.task_params.get('num_of_groups', 3)
        from_to_ids = create_from_to_ids(num_of_groups)
        params = Params(data_name, graph_name, graph_id, graph_path, graph_files, gt_file, num_of_groups, from_to_ids)
        graphs_params.append(params)
    graphs_params_sorted = sorted(graphs_params, key=lambda x: x.id)
    return graphs_params_sorted


# def task_destinations(destination, task, data_name, params):
#     destinations = {}
#     string_list = ["task", str(task), data_name, params.get('embedding')]
#     string_list = [string for string in string_list if string]
#     destinations['results'] = SEP.join(destination, "_".join(string_list + ["results"]))
#     destinations['runtime'] = SEP.join(destination, "_".join(string_list + ["runtime.csv"]))
#     destinations['memory'] = SEP.join(destination, "_".join(string_list + ["memory.csv"]))
#     return destinations

# def save_to_file(lines_list, path):
#     with open(path, 'w', newline='') as file:
#         wr = csv.writer(file)
#         for line in lines_list:
#             wr.writerow(line)


def plot_graph(data_paths, colors, x_label, y_label, save_path=None, my_labels=None, title=''):
    x_list, y_list, labels = [], [], []
    for path in data_paths:
        with open(path, 'r', newline='') as file:
            data = list(csv.reader(file))
            x_list.append([float(i) for i in data[0][1:]])
            y_list.append([float(i) for i in data[1][1:]])
            labels.append(data[1][0])
    if my_labels:
        labels = my_labels
    for x, y, color, label in zip(x_list, y_list, colors, labels):
        x, y = zip(*sorted(zip(x, y)))
        plt.plot(x, y, "-ok", color=color, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xticks(np.arange(0.0, 0.6, 0.1))
    # plt.xscale('log')
    if y_label != 'Accuracy':
        plt.yscale('log')
    plt.title(title)
    plt.grid(True, linestyle='--', which="both")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()


def plot_results(results, colors, x_label, y_label, graph_name, labels=None, title=''):
    paths = []
    for result in results:
        path = SEP.join(["..",
                            "Results",
                            "_".join(["task", result["task"]]),
                            "_".join(["task", result["task"], result["name"]]),
                            "_".join(["task", result["task"], result["name"], result["graph"]]) + '.csv'])
        paths.append(path)
    save_path = SEP.join(["..", "Results", graph_name + ".png"])
    plot_graph(paths, colors, x_label=x_label, y_label=y_label, save_path=save_path, my_labels=labels, title=title)
    # plot_graph(paths, colors, x_label="Fraction", y_label="Running Time[s]", save_path=save_path)
    # plot_graph([memory_destination], ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")


def plot_all_results():
    # plot_graph(runtime_destination, ["blue"], x_label="Fraction", y_label="Running Time[s]",
    #            save_path=runtime_destination[:-4] + ".png")
    # plot_graph(memory_destination, ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")
    # runtime_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_runtime.csv'
    # memory_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_memory.csv'

    results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_avg_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'ogre_avg_accuracy'}]
    plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_accuracy',
                 ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass avg accuracy')

    results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_avg_norm_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'ogre_avg_norm_accuracy'}]
    plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
                 ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass avg norm accuracy')

    results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_winner_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'ogre_winner_accuracy'}]
    plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
                 ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass winner accuracy')

    results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_top5_accuracy'},
               {'task': '4', 'name': 'false_mass', 'graph': 'ogre_top5_accuracy'}]
    plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
                 ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass top-5 accuracy')

    # ---
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_avg_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes avg accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_avg_norm_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_winner_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes winner accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_top5_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes top-5 accuracy')
    # ---
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass winner accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass top-5 accuracy')
    # ---
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass winner accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass top-5 accuracy')

    # results = [{'task': '1', 'name': 'false_mass', 'graph': 'runtime'}]
    # colors = ['blue']
    # plot_results(results, colors, 'Fraction', 'Running Time[s]', 'task_1_false_mass_nodes_runtime')

    # results = [{'task': '1', 'name': 'false_mass', 'graph': 'memory'}]
    # colors = ['blue']
    # plot_results(results, colors, 'Fraction', 'Memory Usage[Mb]', 'task_1_false_mass_memory')





# def eval_destinations(destination, task, data_name, methods, params):
#     destinations = {}
#     string_list = ["task", task, data_name, params.get('embedding')]
#     string_list = [string for string in string_list if string]
#     destinations['results'] = SEP.join([destination, "_".join(string_list + ["results"])])
#     for method in methods:
#         destinations[method] = SEP.join([destination, "_".join(string_list + [method, "accuracy.csv"])])
#     return destinations

def create_from_to_ids(num_of_groups):
    # this function create something like this: [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    from_to_ids = []
    for i in range(num_of_groups):
        for j in range(num_of_groups):
            if i < j:
                from_to_ids.append((i, j))
                from_to_ids.append((j, i))
    return from_to_ids



def run(task, graphs_params, evaluate=True):
    print("Running task", task, 'on data', graphs_params[0].data_name, 'with params', task.task_params)
    task.clean()
    memory_list = []
    for graph_params in graphs_params:
        # if int(graph_name.split('_')[0]) != 300:
        #     continue
        # print("TESTING MODE!!")

        # if task == '1':
        #     graphs_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
        #     results_files = [os.path.join(results_dir, file_name) for file_name in os.listdir(graph_path)]
        #     args = (graph_files, results_files, graphs_params)
        #
        # elif task == '2':
        #     # TODO: make it more pretty
        #     if lol:
        #         graph = MultipartiteLol()
        #         graph.convert_with_csv(graph_files, from_to_ids)
        #         graph.set_nodes_type_dict()
        #     else:
        #         graph = load_graph_from_files(graph_files, from_to_ids, has_title=True, cutoff=0.0)
        #     results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
        #     args = (graph, results_file, 0., [10., 10., 10.])
        #     args_dict = {'assess': False, 'ground_truth': None, 'draw': False}
        #
        # elif task == '3':
        #     max_steps = 4
        #     starting_points = 5  # '0_2'
        #     results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
        #     args = (max_steps, starting_points, graph_files, from_to_ids, results_file)
        #
        # elif task == '4':
        #     # TODO: change this to match embedding
        #     results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
        #     args = (graph_files, results_file, from_to_ids, num_of_groups)
        #     args_dict = {param: graphs_params[param] for param in ['embedding', 'epsilon'] if param in graphs_params}

        memory = memory_usage((task.run, (graph_params,)), max_iterations=1)
        memory_list.append(max(memory))
        task.save_attributes(memory_list)

    if evaluate:
        eval(task, graphs_params)



def eval(task, graphs_params, methods=None):
    print("Evaluating task", task, 'on data', graphs_params[0].data_name)
    task.clean()
    if not methods:
        methods = task.eval_methods

    # if not methods:
    #     if task == '1':
    #         methods = ['avg', 'winner', 'top5']
    #     elif task == '2':
    #         methods = ['avg_full', 'avg_all']
    #     elif task == '3':
    #         methods = ['avg', 'avg_norm', 'winner', 'top5']
    #     elif task == '4':
    #         methods = ['avg', 'avg_norm', 'winner', 'top5']
    #     else:
    #         raise Exception('task', task, 'not found')

    # destinations = eval_destinations(task.destination, task, data_name, methods, params)
    # # results_destination = os.path.join(location, "_".join(["task", task, data_name, "results"]))
    # if not os.path.exists(destinations['results']):
    #     raise Exception("results not found")
    # results_dirs = [os.path.join(destinations['results'], results_dir) for results_dir in
    #                 os.listdir(destinations['results'])]
    # if len(results_dirs) == 0:
    #     raise Exception("results are empty")
    # graphs_names = ['_'.join(graph_dir.split('_')[2:-1]) for graph_dir in os.listdir(destinations['results'])]


    for method in methods:
        print("Evaluating task", str(task), 'on data', graphs_params[0].data_name, 'with method', method)
        # accuracy_list = []
        # x_list = []
        # accuracy_destination = os.path.join(location, "_".join(["task", task, data_name, method, "accuracy.csv"]))
        task.clean()
        for graph_params in graphs_params:
            # print("Evaluating graph", graph_name, "with method", method)
            # results_files = [os.path.join(results_dir, results_file) for results_file in os.listdir(results_dir)]
            task.eval(graph_params, method)
        task.save_eval(method)

def run_task(task_num, data_name, results_root, task_params, evaluate=True):
    if task_num == '1':
        task = BipartiteProbabilisticMatchingTask(results_root, task_params=task_params)
    elif task_num == '2':
        task = MultipartiteCommunityDetectionTask(results_root, task_params=task_params)
    elif task_num == '3':
        task = PathwayProbabilitiesCalculationTask(results_root, task_params=task_params)
    elif task_num == '4':
        task = ProbabilitiesUsingEmbeddingsTask(results_root, task_params=task_params)
    else:
        raise Exception("task_num need to be between 1 and 4")
    graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    run(task=task, graphs_params=graphs_params, evaluate=evaluate)

if __name__ == '__main__':
    # FOR ACCURATE RUNTIME, ALWAYS RUN ON NORMAL, NOT DEBUG!
    random.seed(0)
    np.random.seed(0)
    if 'Code' not in os.listdir(os.getcwd()):
        raise Exception("Bad pathing, use the command os.chdir() to make sure you work on Magnet directory")
    start = time.time()

    # Long version run:
    #
    # results_root = "Results"
    # task_params = {'num_of_groups': 3}
    # data_name = 'false_mass'
    # task = BipartiteProbabilisticMatchingTask(results_root, task_params=task_params)
    # graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    # run(task=task, graphs_params=graphs_params)




    # examples for 2 groups:
    # run_task(task_num="1", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num="2", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num="3", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num='4', data_name='test', results_root='Results', task_params={'num_of_groups': 2, 'embedding': 'node2vec'})
    # run_task(task_num='4', data_name='test', results_root='Results', task_params={'num_of_groups': 2, 'embedding': 'ogre', 'epsilon': 0.1})

    # examples for 3 groups:
    run_task(task_num="1", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    run_task(task_num="2", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    run_task(task_num="3", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    run_task(task_num='4', data_name='removed_nodes', results_root='Results', task_params={'num_of_groups': 3, 'embedding': 'node2vec'})
    run_task(task_num='4', data_name='removed_nodes', results_root='Results', task_params={'num_of_groups': 3, 'embedding': 'ogre', 'epsilon': 0.1})


    # plot_all_results()
    print('TOTAL TIME:', time.time() - start)
