import argparse
import os
from PathwayProbabilitiesCalculation.code.pathway_probabilities_calculation import task3
from PathwayProbabilitiesCalculation.code.probabilities_using_embeddings import task4
import matplotlib.pyplot as plt
import matplotlib
from BipartiteProbabilisticMatching.code.matching_solutions import MatchingProblem, task1, eval_task1

import time
import numpy as np
import cProfile
from memory_profiler import memory_usage
import csv

# from Tests.NoisyEdges.noisy_edges_test import *
from Tests.RemovedRealVertices.removed_real_vertices_test import *
from multipartite_lol_graph import MultipartiteLol

lol = True
if lol:
    from MultipartiteCommunityDetection.code.run_louvain_lol import run_louvain, task2, eval_task2
else:
    from MultipartiteCommunityDetection.code.run_louvain import run_louvain, task2, load_graph_from_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-T")
    parser.add_argument("--source", "-S")
    parser.add_argument("--destination", "-D")
    args = parser.parse_args()
    task_number = args.task
    source_dir = args.source
    destination_dir = args.destination

    if task_number == '1' and source_dir and destination_dir:
        rootDir = os.path.join("..", "BipartiteProbabilisticMatching", "data", source_dir)
        graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                            os.walk(rootDir) for file in filenames]

        first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                                 destination_dir,
                                                 f"yoram_network_graph_{g}.csv") for g in range(1, 4)]
        first_stage_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}

        if not os.path.exists(os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                           destination_dir)):
            os.makedirs(os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                     destination_dir))

        if len(graph_file_names) == 0:
            print("No file were found!")
        else:
            task1(graph_file_names, first_stage_saving_paths, first_stage_params)

    elif task_number == '2' and source_dir and destination_dir:
        lol = True

        from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]

        rootDir = os.path.join("..", "MultipartiteCommunityDetection", "data", source_dir)
        second_graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                                  os.walk(rootDir) for file in filenames]

        if lol:
            from MultipartiteCommunityDetection.code.run_louvain_lol import run_louvain, task2
            graph = MultipartiteLol()
            graph.convert_with_csv(second_graph_filenames, from_to_ids)
            graph.set_nodes_type_dict()
            destination_dir = destination_dir + "_lol"
        else:
            from MultipartiteCommunityDetection.code.run_louvain import run_louvain, task2, load_graph_from_files
            graph = load_graph_from_files(second_graph_filenames, from_to_ids, has_title=True, cutoff=0.0)
            destination_dir = destination_dir + "networkx"

        task2(graph, destination_dir, 0., [10., 10., 10.], assess=False, ground_truth=None, draw=False)

    elif task_number == '3' and source_dir and destination_dir:
        max_steps = 4
        starting_point = '0_2'
        rootDir = os.path.join("..", "PathwayProbabilitiesCalculation", "data", source_dir)
        graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                            os.walk(rootDir) for file in filenames]
        graph_file_names.sort()  # optional

        from_to_groups = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
        task3(max_steps, starting_point, graph_file_names, from_to_groups, destination_dir)

    else:
        print("wrong/missing arguments...")



# def for_memory_task1(directory, idx):
#     rootDir = directory
#     graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
#                         os.walk(rootDir) for file in filenames]
#     graph_file_names.sort()
#     destination_dir = os.path.join(
#         os.path.join("..", "BipartiteProbabilisticMatching", "results", "experiment_result" + str(idx)))
#     first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
#                                 "experiment_result" + str(idx),
#                                              f"yoram_network_1_graph_{g}.csv") for g in range(1, 4)]
#     first_stage_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
#
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
#
#     task1(graph_file_names, first_stage_saving_paths, first_stage_params)
#
#
# def for_memory_task2(directory, idx, lol_flag):
#     from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
#     rootDir = directory
#     graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
#                         os.walk(rootDir) for file in filenames]
#     graph_file_names.sort()
#
#     if lol_flag:
#         from MultipartiteCommunityDetection.code.run_louvain_lol import run_louvain, task2
#         graph = MultipartiteLol()
#         graph.convert_with_csv(graph_file_names, from_to_ids)
#         graph.set_nodes_type_dict()
#         print("Number of edges", graph.number_of_edges())
#         print("Number of nodes", graph.number_of_nodes())
#         destination_dir = "experiment_result_lol" + str(idx)
#     else:
#         from MultipartiteCommunityDetection.code.run_louvain import run_louvain, task2, load_graph_from_files
#         graph = load_graph_from_files(graph_file_names, from_to_ids, has_title=True, cutoff=0.0)
#         destination_dir = "experiment_result_networkx" + str(idx)
#
#     task2(graph, destination_dir, 0., [10., 10., 10.], assess=False, ground_truth=None, draw=False)
#
# # def ogre(directory, idx, lol_flag):
# #     from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
# #     rootDir = directory
# #     graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
# #                         os.walk(rootDir) for file in filenames]
# #     graph_file_names.sort()
# #     from MultipartiteCommunityDetection.code.run_louvain import run_louvain, task2, load_graph_from_files
# #     graph = load_graph_from_files(graph_file_names, from_to_ids, has_title=True, cutoff=0.0)
# #     destination_dir = "experiment_result_networkx" + str(idx)
# #
# #     task2(graph, destination_dir, 0., [10., 10., 10.], assess=False, ground_truth=None, draw=False)
#
#
# def check_memory_usage_task1():
#     rootDir = os.path.join("..", "BipartiteProbabilisticMatching", "graph_experiment")
#
#     nodes_size = [500, 2500, 5000, 10000, 15000, 20000, 25000]
#     dirctory_names = [rootDir + "/graph" + str(idx) for idx in nodes_size]
#     memory_list = []
#     timer_list = []
#     for idx, directory in enumerate(dirctory_names):
#         print(directory)
#         start = time.time()
#         k = memory_usage((for_memory_task1, (directory, idx)))
#         end = time.time()
#         timer_list.append(end-start)
#         memory_list.append(max(k))
#         print("The memory of task1 is: ", k)
#         print(f"Task 1 took {end - start} s")
#         k.clear()
#     plt.plot(nodes_size, memory_list, "-ok", color='red')
#     plt.xlabel("Size")
#     plt.ylabel("Memory Usage [MB]")
#     plt.title("Memory Usage As Function Of Size")
#     plt.savefig("Memory_Usage_Task1.png")
#
#     plt.clf()
#
#     plt.plot(nodes_size, timer_list, "-ok", color='blue')
#     plt.xlabel("Size")
#     plt.ylabel("Time [S]")
#     plt.title("Running time As Function Of Size")
#     plt.savefig("Running_Time_Task1.png")
#
#
# def check_memory_and_time_task2():
#     rootDir = os.path.join("..", "MultipartiteCommunityDetection", "graph_experiment")
#
#     nodes_size = [500, 2500, 5000, 10000, 15000, 20000, 25000]
#     # nodes_size = [500, 2500, 5000]
#     dirctory_names = [rootDir + "/experiment_result" + str(idx) for idx in range(len(nodes_size))]
#     memory_list_lol = []
#     timer_list_lol = []
#
#     memory_list_networkx = []
#     timer_list_networkx = []
#     for idx, directory in enumerate(dirctory_names):
#         print(directory)
#         start = time.time()
#         k = memory_usage((for_memory_task2, (directory, idx, True)))
#         end = time.time()
#         timer_list_lol.append(end - start)
#         memory_list_lol.append(max(k))
#         # print("The memory of task2 LOL is: ", k)
#         print(f"Task 2(Lol) took {end - start} s")
#         k.clear()
#
#         start = time.time()
#         k = memory_usage((for_memory_task2, (directory, idx, False)))
#         end = time.time()
#         timer_list_networkx.append(end - start)
#         memory_list_networkx.append(max(k))
#         # print("The memory of task2 NETWORKX is: ", k)
#         print(f"Task 2(Networkx) took {end - start} s")
#         k.clear()
#
#     plt.plot(nodes_size, memory_list_lol, "-ok", color='red')
#     plt.xlabel("Size")
#     plt.ylabel("Memory Usage [MB]")
#     plt.title("Memory Usage As Function Of Size(Lol)")
#     plt.savefig("Memory_Usage_Louvain_Like_Lol.png")
#
#     plt.clf()
#
#     plt.plot(nodes_size, timer_list_lol, "-ok", color='blue')
#     plt.xlabel("Size")
#     plt.ylabel("Time [s]")
#     plt.title("Running time As Function Of Size (Lol)")
#     plt.savefig("Running_Time_Louvain_Like_Lol.png")
#
#     plt.clf()
#
#     plt.plot(nodes_size, memory_list_networkx, "-ok", color='red')
#     plt.xlabel("Size")
#     plt.ylabel("Memory Usage [MB]")
#     plt.title("Memory Usage As Function Of Size (Networkx)")
#     plt.savefig("Memory_Usage_Louvain_Like_Networkx.png")
#
#     plt.clf()
#
#     plt.plot(nodes_size, timer_list_networkx, "-ok", color='blue')
#     plt.xlabel("Size")
#     plt.ylabel("Time [s]")
#     plt.title("Running time As Function Of Size (Networkx)")
#     plt.savefig("Running_Time_Louvain_Like_Networkx.png")


if __name__ == '__main__':
    print()
