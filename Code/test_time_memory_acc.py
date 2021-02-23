import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import cProfile
from memory_profiler import memory_usage
import os
from multipartite_lol_graph import MultipartiteLol

"""
I think this is old
"""

def for_memory_task1(directory, idx):
    rootDir = directory
    graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                        os.walk(rootDir) for file in filenames]
    graph_file_names.sort()
    destination_dir = os.path.join(
        os.path.join("..", "BipartiteProbabilisticMatching", "results", "added_false_edge_result" + str(idx)))
    first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                "added_false_edge_result" + str(idx),
                                             f"added_false_edges_frc_{idx}_graph_{g}.csv") for g in range(1, 4)]
    first_stage_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    task1(graph_file_names, first_stage_saving_paths, first_stage_params)


def for_memory_task2(directory, idx, lol_flag):
    from_to_ids = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
    rootDir = directory
    graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                        os.walk(rootDir) for file in filenames]
    graph_file_names.sort()

    if lol_flag:
        from MultipartiteCommunityDetection.code.run_louvain_lol import run_louvain, task2
        graph = MultipartiteLol()
        graph.convert_with_csv(graph_file_names, from_to_ids)
        graph.set_nodes_type_dict()
        print("Number of edges", graph.number_of_edges())
        print("Number of nodes", graph.number_of_nodes())
        destination_dir = "experiment_result_lol" + str(idx)
    else:
        from MultipartiteCommunityDetection.code.run_louvain import run_louvain, task2, load_graph_from_files
        graph = load_graph_from_files(graph_file_names, from_to_ids, has_title=True, cutoff=0.0)
        destination_dir = "experiment_result_networkx" + str(idx)

    task2(graph, destination_dir, 0., [10., 10., 10.], assess=False, ground_truth=None, draw=False)


def check_memory_usage_task1(x_axis, dirctory_names):
    # rootDir = os.path.join("..", "BipartiteProbabilisticMatching", "graph_experiment_nodes_number")
    #
    # nodes_size = [500, 2500, 5000, 10000, 15000, 20000, 25000]
    # dirctory_names = [rootDir + "/graph" + str(idx) for idx in nodes_size]
    memory_list = []
    timer_list = []
    for idx, directory in enumerate(dirctory_names):
        print("---------------------")
        print(directory)
        start = time.time()
        k = memory_usage((for_memory_task1, (directory, x_axis[idx])))
        end = time.time()
        timer_list.append(end-start)
        memory_max = max(k)
        memory_list.append(memory_max)
        print(f"The memory of Task 1 is: {memory_max}, MB")
        print(f"Task 1 took {end - start} s")
        k.clear()

    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, memory_list, "-ok", color='red')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Memory Usage")
    plt.xscale('log')
    plt.yscale('log')
    ax1.set_xticks(x_axis)
    ax1.set_yticks(memory_list)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Memory Usage As Function Of Noisy Edges Fraction")
    plt.savefig("Memory_Usage_Task_Normalization_Noisy_Edges_Fraction.png")

    plt.clf()

    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, timer_list, "-ok", color='blue')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Time")
    plt.xscale('log')
    plt.yscale('log')
    ax1.set_xticks(x_axis)
    ax1.set_yticks(timer_list)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Running time As Function Of Noisy Edges Fraction")
    plt.savefig("Running_Time_Task_Normalization_Noisy_Edges_Fraction.png")


def check_memory_and_time_task2(x_axis, dirctory_names):
    # rootDir = os.path.join("..", "MultipartiteCommunityDetection", "graph_experiment_nodes_number")
    #
    # nodes_size = [500, 2500, 5000, 10000, 15000, 20000, 25000]
    # # nodes_size = [500, 2500, 5000]
    # dirctory_names = [rootDir + "/experiment_result" + str(idx) for idx in range(len(nodes_size))]
    memory_list_lol = []
    timer_list_lol = []

    memory_list_networkx = []
    timer_list_networkx = []
    for idx, directory in enumerate(dirctory_names):
        print(directory)
        start = time.time()
        k = memory_usage((for_memory_task2, (directory, idx, True)))
        end = time.time()
        timer_list_lol.append(end - start)
        memory_max = max(k)
        memory_list_lol.append(memory_max)
        print(f"The memory of Task 1 is: {memory_max} MB")
        print(f"Task 2(Lol) took {end - start} s")
        k.clear()

        start = time.time()
        k = memory_usage((for_memory_task2, (directory, idx, False)))
        end = time.time()
        timer_list_networkx.append(end - start)
        memory_max = max(k)
        memory_list_networkx.append(memory_max)
        print(f"The memory of Task 1 is: {memory_max} MB")
        print(f"Task 2(Networkx) took {end - start} s")
        k.clear()

    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, memory_list_lol, "-ok", color='red')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Memory Usage")
    plt.xscale('log')
    plt.yscale('log')
    ax1.set_xticks(x_axis)
    # ax1.set_yticks(memory_list_lol)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Memory Usage As Function Of Noisy Edges Fraction")
    plt.savefig("Memory_Usage_Louvain_Noisy_Edges_MassLol.png")

    plt.clf()

    fig1, ax1 = plt.subplots()
    plt.plot(x_axis, timer_list_lol, "-ok", color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Time")
    ax1.set_xticks(x_axis)
    # ax1.set_yticks(timer_list_lol)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Running time As Function Of Noisy Edges Fraction")
    plt.savefig("Running_Time_Louvain_Noisy_Edges_Mass_Lol.png")

    plt.clf()

    fig1, ax1 = plt.subplots()
    plt.plot(x_axis, memory_list_networkx, "-ok", color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Memory Usage")
    ax1.set_xticks(x_axis)
    # ax1.set_yticks(memory_list_networkx)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Memory Usage As Function Of Noisy Edges Fraction")
    plt.savefig("Memory_Usage_Louvain_Noisy_Edges_Fraction_Networkx.png")

    plt.clf()

    fig1, ax1 = plt.subplots()
    plt.plot(x_axis, timer_list_networkx, "-ok", color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Noisy Edges Fraction")
    plt.ylabel("Time")
    ax1.set_xticks(x_axis)
    # ax1.set_yticks(timer_list_networkx)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Running time As Function Of Noisy Edges Fraction")
    plt.savefig("Running_Time_Louvain_Noisy_Edges_Fraction_Networkx.png")


def check_accuracy(partition):
    coms = {}
    lst = []
    #print("len(partition)",len(partition))
    for v, c in partition.items():
        lst.append(v)
        if c not in coms.keys():
            coms[c] = [v]
        else:
            coms[c].append(v)
    counts = {1: 0, 2: 0, 3: 0}
    counts_good = 0
    for c, lst1 in coms.items():
        # print(len(lst), lst)
        counts[len(lst1)] += 1
        if len(lst1) == 3 and lst1[0].split("_")[1] == lst1[1].split("_")[1] == lst1[2].split("_")[1]:
            counts_good += 1
    return counts_good/float(counts[3]) * 100


def calc_task2_acc(files):
    # nodes_size = [500, 2500, 5000, 10000, 15000, 20000, 25000]
    acc_list = []
    for file in files:
        partition = {}
        with open(file, "r") as file:
            file.readline()
            for line in file:
                split_line = line.split(',')
                split_line[2] = split_line[2].replace("\n", "")
                partition[split_line[1]+'_'+split_line[0]] = int(split_line[2])
        acc = check_accuracy(partition)
        acc_list.append(acc)
    return acc_list

    # plt.plot(nodes_size, acc_list, "-ok", color = 'green')
    # plt.xlabel("#Nodes")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy As Function Of #Nodes")
    # plt.savefig("acc-Task2_Function_Of_Nodes.png")
