import os
import time
import sys
sys.path.append(os.path.abspath('..'))
from PathwayProbabilitiesCalculation.code.pathway_probabilities_calculation import task3, eval_task3
from PathwayProbabilitiesCalculation.code.probabilities_using_embeddings import task4, eval_task4
import matplotlib.pyplot as plt
from BipartiteProbabilisticMatching.code.matching_solutions import task1, eval_task1

import numpy as np
from memory_profiler import memory_usage
import csv

from multipartite_lol_graph import MultipartiteLol


lol = True
if lol:
    from MultipartiteCommunityDetection.code.run_louvain_lol import task2, eval_task2
else:
    from MultipartiteCommunityDetection.code.run_louvain import task2, load_graph_from_files



def get_data_path(task, data_name):
    # TODO: change data_path to be more dynamic
    if task == '1':
        data_path = os.path.join("..", "Results", "data", data_name)
    else:
        data_path = os.path.join("..", "Results", "task_1", "task_1_" + data_name, "task_1_" + data_name + "_results")
    if not os.path.exists(data_path):
        raise Exception("data not found")
    return data_path


def get_graphs_paths(data_path):
    graphs_names = ['_'.join(dir_name.split('_')[2:-1]) if "_results" in dir_name else dir_name for dir_name in
                    os.listdir(data_path)]
    graphs_paths = [os.path.join(data_path, graph_dir) for graph_dir in os.listdir(data_path)]
    return graphs_paths, graphs_names


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


def save_to_file(lines_list, path):
    with open(path, 'w', newline='') as file:
        wr = csv.writer(file)
        for line in lines_list:
            wr.writerow(line)


def plot_results(results, colors, x_label, y_label, graph_name, labels=None, title=''):
    paths = []
    for result in results:
        path = os.path.join("..",
                            "Results",
                            "_".join(["task", result["task"]]),
                            "_".join(["task", result["task"], result["name"]]),
                            "_".join(["task", result["task"], result["name"], result["graph"]]) + '.csv')
        paths.append(path)
    save_path = os.path.join("..", "Results", graph_name + ".png")
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

def task_destinations(destination, task, data_name, params):
    destinations = {}
    string_list = ["task", task, data_name, params.get('embedding')]
    string_list = [string for string in string_list if string]
    destinations['results'] = os.path.join(destination, "_".join(string_list + ["results"]))
    destinations['runtime'] = os.path.join(destination, "_".join(string_list + ["runtime.csv"]))
    destinations['memory'] = os.path.join(destination, "_".join(string_list + ["memory.csv"]))
    return destinations

def eval_destinations(destination, task, data_name, methods, params):
    destinations = {}
    string_list = ["task", task, data_name, params.get('embedding')]
    string_list = [string for string in string_list if string]
    destinations['results'] = os.path.join(destination, "_".join(string_list + ["results"]))
    for method in methods:
        destinations[method] = os.path.join(destination, "_".join(string_list + [method, "accuracy.csv"]))
    return destinations


def run_task(task, data_name, evaluate=True, params=None):
    print("Running task", task, 'on data', data_name, 'with params', params)
    if params is None:
        params = {}
    func_dict = {'1': task1, '2': task2, '3': task3, '4': task4}
    task_func = func_dict[task]
    data_path = get_data_path(task, data_name)
    graphs_paths, graphs_names = get_graphs_paths(data_path)

    destination = os.path.join("..",
                               "Results",
                               "_".join(["task", task]),
                               "_".join(["task", task, data_name]))

    destinations = task_destinations(destination, task, data_name, params)

    # TODO: change this to match embedding
    results_dirs = [os.path.join(destinations['results'], "_".join(["task", task, graph_name, "results"])) for graph_name in graphs_names]

    runtime_list = []
    memory_list = []
    x_list = []
    args = ()
    args_dict = {}

    # TODO: get rid of if task == i
    for graph_path, results_dir, graph_name in zip(graphs_paths, results_dirs, graphs_names):
        print('Running on graph', graph_name)

        # if int(graph_name.split('_')[0]) != 300:
        #     continue
        # print("TESTING MODE!!")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        graph_files = [os.path.join(graph_path, file_name) for file_name in os.listdir(graph_path)]

        if task == '1':
            params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
            results_files = [os.path.join(results_dir, file_name) for file_name in os.listdir(graph_path)]
            args = (graph_files, results_files, params)

        elif task == '2':
            # TODO: make it more pretty
            from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
            if lol:
                graph = MultipartiteLol()
                graph.convert_with_csv(graph_files, from_to_ids)
                graph.set_nodes_type_dict()
            else:
                graph = load_graph_from_files(graph_files, from_to_ids, has_title=True, cutoff=0.0)
            results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
            args = (graph, results_file, 0., [10., 10., 10.])
            args_dict = {'assess': False, 'ground_truth': None, 'draw': False}

        elif task == '3':
            max_steps = 4
            starting_points = 500  # '0_2'
            from_to_groups = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
            results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
            args = (max_steps, starting_points, graph_files, from_to_groups, results_file)

        elif task == '4':
            from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
            # TODO: change this to match embedding
            results_file = os.path.join(results_dir, "_".join(["task", task, graph_name, "results"]) + '.csv')
            args = (graph_files, results_file, from_to_ids)
            args_dict = params

        memory, runtime = memory_usage((task_func, args, args_dict), retval=True, max_iterations=1)
        runtime_list.append(runtime)
        memory_list.append(max(memory))
        x_list.append(graph_name.split("_")[0].replace(",", "."))

    save_to_file([['x'] + x_list, [data_name] + runtime_list], destinations['runtime'])
    save_to_file([['x'] + x_list, [data_name] + memory_list], destinations['memory'])
    if evaluate:
        eval_task(task, data_name, params)
    print()


def eval_task(task, data_name, params, methods=None, location=None):
    print("Evaluating task", task, 'on data', data_name)
    func_dict = {'1': eval_task1, '2': eval_task2, '3': eval_task3, '4': eval_task4}
    eval_func = func_dict[task]
    if not location:
        location = os.path.join("..",
                                "Results",
                                "_".join(["task", task]),
                                "_".join(["task", task, data_name]))

    if not methods:
        if task == '1':
            methods = ['avg', 'winner', 'top5']
        elif task == '2':
            methods = ['avg_3', 'avg_all']
        elif task == '3':
            methods = ['avg', 'avg_norm', 'winner', 'top5']
        elif task == '4':
            methods = ['avg', 'avg_norm', 'winner', 'top5']
        else:
            raise Exception('task', task, 'not found')

    destinations = eval_destinations(location, task, data_name, methods, params)
    # results_destination = os.path.join(location, "_".join(["task", task, data_name, "results"]))
    if not os.path.exists(destinations['results']):
        raise Exception("results not found")
    results_dirs = [os.path.join(destinations['results'], results_dir) for results_dir in
                    os.listdir(destinations['results'])]
    if len(results_dirs) == 0:
        raise Exception("results are empty")
    graphs_names = ['_'.join(graph_dir.split('_')[2:-1]) for graph_dir in os.listdir(destinations['results'])]

    for method in methods:
        accuracy_list = []
        x_list = []
        # accuracy_destination = os.path.join(location, "_".join(["task", task, data_name, method, "accuracy.csv"]))
        for results_dir, graph_name in zip(results_dirs, graphs_names):
            print(graph_name)
            results_files = [os.path.join(results_dir, results_file) for results_file in os.listdir(results_dir)]
            accuracy = eval_func(results_files, method)
            accuracy_list.append(accuracy)
            x_list.append(graph_name.split("_")[0].replace(",", "."))
        save_to_file([['x'] + x_list, [data_name] + accuracy_list], destinations[method])

if __name__ == '__main__':
    # FOR ACCURATE RUNTIME, ALWAYS RUN ON NORMAL, NOT DEBUG!
    start = time.time()
    np.random.seed(42)
    # eval_task(task='4', data_name='nodes', params={'embedding': 'ogre'})
    # run_task(task="1", data_name="removed_nodes")
    # run_task(task="2", data_name="false_mass")
    # run_task(task="3", data_name="false_mass", evaluate=False)
    # run_task(task="4", data_name="noisy_edges", params={'embedding': 'ogre'})
    # run_task(task="3", data_name="nodes", evaluate=False)
    # run_task(task="4", data_name="false_mass", evaluate=True)
    # run_task(task="1", data_name="nodes")
    # eval_task('3', 'removed_nodes')
    # eval_task('4', 'false_mass')
    # for task in ['1', '2', '3', '4']:
    #     for data_name in ['false_mass', 'nodes', 'noisy_edges', 'removed_nodes']:
    #         run_task(task=task, data_name=data_name)

    # run_task(task='4', data_name='false_mass', params={'embedding': 'ogre', 'epsilon': 0.1})
    # run_task(task='4', data_name='false_mass', params={'embedding': 'node2vec'})
    # run_task(task='4', data_name='nodes', params={'embedding': 'ogre', 'epsilon': 0.1})
    # run_task(task='4', data_name='nodes', params={'embedding': 'node2vec'})
    # plot_all_results()
    print('TOTAL TIME:',time.time() - start)
