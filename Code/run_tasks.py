# This version of MAGNET contains all the features from the thesis. BUT this is not a very user friendly version.
# To make it organized and use friendly, more work need to be done.


import os
import sys

from Code.graphs import Params
from Code.tasks import BipartiteProbabilisticMatchingTask, MultipartiteCommunityDetectionTask, \
    PathwayProbabilitiesCalculationTask, ProbabilitiesUsingEmbeddingsTask, BipartiteNaiveTask, MultipartiteGreedyTask

sys.path.append(os.path.abspath('..'))

import numpy as np
from memory_profiler import memory_usage
SEP = '/'

"""
#### README ####
This file is just functions that help us run the tasks.
There is no logic in here, mainly:
 - Paths manipulation (the results dir is very complex)
 - Here we make sure the task 1 take his data from Results/data, and the rest take it from Results/task_1
 - Initialize some objects
 - Grid search and full runs
 - Examples
"""

def get_data_path(task, results_root, data_name):
    if task.use_task_1_results:
        data_path = SEP.join([results_root, 'task_1_BipartiteProbabilisticMatching', data_name, 'results'])
    else:
        data_path = SEP.join([results_root, 'data', data_name])
    if not os.path.exists(data_path):
        raise Exception(f"data '{data_name}' not found")
    return data_path


def get_graphs_paths(data_path):
    graphs_paths = [SEP.join([data_path, graph_dir]) for graph_dir in os.listdir(data_path)]
    # graphs_names = ['_'.join([param_str] + dir_name.split('_')[:-1]) if "_results" in dir_name else param_str + '_' + dir_name for dir_name in
    #                 os.listdir(data_path)]
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
        if not task.use_task_1_results and len(graph_files) != num_of_groups * (num_of_groups - 1) / 2 or \
           task.use_task_1_results and len(graph_files) != num_of_groups * (num_of_groups - 1):
            raise Exception("num_of_groups seems to be wrong")
        from_to_ids = create_from_to_ids(num_of_groups)
        params = Params(data_name, graph_name, graph_id, graph_path, graph_files, gt_file, num_of_groups, from_to_ids)
        graphs_params.append(params)
    graphs_params_sorted = sorted(graphs_params, key=lambda x: x.id)
    return graphs_params_sorted



def create_from_to_ids(num_of_groups):
    # this function create something like this: [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    from_to_ids = []
    for i in range(num_of_groups):
        for j in range(num_of_groups):
            if i < j:
                from_to_ids.append((i, j))
                from_to_ids.append((j, i))
    return from_to_ids

def grid_search(data_name):
    best_score = 0
    best_params = None
    results_root = "Results"
    for rho_0 in np.round(np.arange(0.8, 1, 0.02), 2):
        for rho_1 in np.round(np.arange(0.0, np.round(1 - rho_0, 2), 0.02), 2):
            for epsilon in np.round(np.arange(0.01, 0.021, 0.005), 3):

                task_params = {'num_of_groups': 2, 'rho_0': rho_0, 'rho_1': rho_1, 'epsilon': epsilon}
                # task_params = {'num_of_groups': 3, 'starting_points': 10000}
                task = BipartiteProbabilisticMatchingTask(results_root, task_params=task_params)
                graphs_params = get_graphs_params(task, results_root, data_name=data_name)
                run(task=task, graphs_params=graphs_params, evaluate=False)
                # eval(task=task, graphs_params=graphs_params)
                for t in np.round(np.arange(0, 0.3, 0.01), 2):  # [0.3, 0.4, 0.5, 0.6]:
                    task_params['f1_threshold'] = t
                    task.task_params = task_params
                    graphs_params = get_graphs_params(task, results_root, data_name=data_name)
                    eval(task=task, graphs_params=graphs_params, methods=['f1_score'])
                    if max(task.scores) > best_score:
                        best_score = max(task.scores)
                        best_params = task_params.copy()
                        print("New best score!")
                        task.save_best('f1_score')
                print('best_score:', best_score)
                print('best_params:', best_params)
    task = BipartiteProbabilisticMatchingTask(results_root, task_params=best_params)
    graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    run(task=task, graphs_params=graphs_params, evaluate=True)


def full_run(beta):
    real_graphs = ['Fodors-Zagats', 'Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'Abt-Buy']
    simulation_graphs = ['false_mass', 'noisy_edges', 'removed_nodes', 'nodes']
    for data_name in simulation_graphs:
        # grid_search(data_name)
        run_task(task_num="1", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
        run_task(task_num="2", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3, 'beta': beta})
        run_task(task_num="3", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
        # Task 4 is not relevant anymore
        # run_task(task_num='4', data_name=data_name, results_root="Results", task_params={'num_of_groups': 2, 'embedding': 'node2vec'})
        run_task(task_num="5", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
        run_task(task_num="6", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})



def run(task, graphs_params, evaluate=True, methods=None):
    print("Running task", task, 'on data', graphs_params[0].data_name, 'with params', task.task_params)
    task.clean()
    memory_list = []
    for graph_params in graphs_params:
        # if int(graph_name.split('_')[0]) != 300:
        #     continue
        # print("TESTING MODE!!")

        memory = memory_usage((task.run, (graph_params,)), max_iterations=1)
        memory_list.append(max(memory))
        task.save_attributes(memory_list)

    if evaluate:
        eval(task, graphs_params, methods)



def eval(task, graphs_params, methods=None):
    print("Evaluating task", task, 'on data', graphs_params[0].data_name, 'with params', task.task_params)
    task.clean()
    if not methods:
        methods = task.eval_methods

    for method in methods:
        print('Method:', method)
        task.clean()
        for graph_params in graphs_params:
            task.eval(graph_params, method)
        task.save_eval(method)
        print("Scores:", task.scores, "Best:", max(task.scores))


def run_task(task_num, data_name, results_root, task_params, evaluate=True):
    if task_num == '1':
        task = BipartiteProbabilisticMatchingTask(results_root, task_params=task_params)
    elif task_num == '2':
        task = MultipartiteCommunityDetectionTask(results_root, task_params=task_params)
    elif task_num == '3':
        task = PathwayProbabilitiesCalculationTask(results_root, task_params=task_params)
    elif task_num == '4':
        task = ProbabilitiesUsingEmbeddingsTask(results_root, task_params=task_params)
    elif task_num == '5':
        task = BipartiteNaiveTask(results_root, task_params=task_params)
    elif task_num == '6':
        task = MultipartiteGreedyTask(results_root, task_params=task_params)
    else:
        raise Exception("task_num need to be between 1 and 6")
    graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    run(task=task, graphs_params=graphs_params, evaluate=evaluate)
    # eval(task=task, graphs_params=graphs_params)

def examples():
    print("Here are some examples!")
    # grid_search()

    # Data names: [false_mass, nodes, noisy_edges, removed_nodes, restaurant, test, Abt-Buy, toy, toy_multi]
    # tasks: [BipartiteProbabilisticMatchingTask, MultipartiteCommunityDetectionTask, BipartiteNaiveTask, MultipartiteGreedyTask]


    ##################################
    # Long version:
    #
    # results_root = "Results"
    # task_params = {'num_of_groups': 2}
    # data_name = 'toy'
    # data_name = 'toy_multi'
    # task = MultipartiteCommunityDetectionTask(results_root, task_params=task_params)
    # graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    # run(task=task, graphs_params=graphs_params, evaluate=False)
    # eval(task=task, graphs_params=graphs_params)
    # for t in np.round(np.arange(0, 0.0, 0.01), 2):  # [0.3, 0.4, 0.5, 0.6]:
    #     task_params['f1_threshold'] = t
    #     task.task_params = task_params
    #     graphs_params = get_graphs_params(task, results_root, data_name=data_name)
    #     eval(task=task, graphs_params=graphs_params, methods=['f1_score'])
    ##################################


    ##################################
    # Short version:
    #
    # examples for 2 groups:
    # run_task(task_num="1", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num="2", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num="3", data_name="test", results_root='Results', task_params={'num_of_groups': 2})
    # run_task(task_num='4', data_name='test', results_root='Results', task_params={'num_of_groups': 2, 'embedding': 'node2vec'})
    # run_task(task_num='4', data_name='test', results_root='Results', task_params={'num_of_groups': 2, 'embedding': 'ogre', 'epsilon': 0.1})
    #
    # examples for 3 groups:
    # run_task(task_num="1", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    # run_task(task_num="2", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    # run_task(task_num="3", data_name="removed_nodes", results_root='Results', task_params={'num_of_groups': 3})
    # run_task(task_num='4', data_name='removed_nodes', results_root='Results', task_params={'num_of_groups': 3, 'embedding': 'node2vec'})
    # run_task(task_num='4', data_name='removed_nodes', results_root='Results', task_params={'num_of_groups': 3, 'embedding': 'ogre', 'epsilon': 0.1})
    ##################################


    ##################################
    # Plots:
    # plot_all_results('results_2.png')
    # plot_all_results_task_2()
    # plot_all_results_task_2_v2()
    # plot_toy_graphs(file_names=graphs_params[0].files, name="before", graphs_directions=[(0, 1)], problem=[4, 9], edge_width=0.14)
    # plot_toy_graphs(file_names=[task.results_files[0]], name="after_01", directed=True, graphs_directions=[(0, 1)], header=True, integer=False, problem=[0.16, 0.79], edge_width=3)
    # plot_toy_graphs(file_names=[task.results_files[1]], name="after_10", directed=True, graphs_directions=[(1, 0)], header=True, integer=False, problem=[0.82, 0.16], edge_width=3)
    #
    # plot_toy_graphs(file_names=graphs_params[0].files, name="multi_0", graphs_directions=[(0, 1), (0, 2), (1, 2)], problem=[])
    # plot_toy_graphs(file_names=graphs_params[0].files, name="multi_1", graphs_directions=[(0, 1), (0, 2), (1, 2)], problem=[], edge_width=1)
    # plot_toy_graphs(file_names=graphs_params[0].files, name="multi_2", graphs_directions=[(0, 1), (0, 2), (1, 2)],problem=[], edge_width=1, colored=True)
    ##################################

if __name__ == '__main__':
    print()


