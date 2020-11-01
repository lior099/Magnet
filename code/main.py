import argparse
import os
import sys
from PathwayProbabilitiesCalculation.code.pathway_probabilities_calculation import task3, load_graph
from MultipartiteCommunityDetection.code.run_louvain import load_graph_from_files, load_ground_truths, run_louvain, task2
from BipartiteProbabilisticMatching.code.matching_solutions import MatchingProblem, task1
import time


def first_second_stage(graph_filenames, graph_ids, algorithm, first_stage_params, first_saving_paths, cutoff,
                       resolution, beta_vec, second_saving_path, assess, draw, community_ground_truth=None):
    """
    Implement both the first (i.e. find probabilities of matching from one node to another for every bipartite
    source-target graph) and second (i.e. from the directed graph with edge weights found earlier, divide the graph into
    communities intended to represent one entity).
    :param graph_filenames: A list of the names of csv files, each of which holds an undirected weighted bipartite
           graph.
    :param graph_ids: A corresponding list of tuples representing the types of source and target nodes.
    :param algorithm: The algorithm to apply in the first stage.
    :param first_stage_params: The hyper-parameters for the algorithm.
    :param first_saving_paths: The paths (a list corresponding to graph_filenames) in which the results of the first
           stage will be saved. Those are full paths.
    :param cutoff: A float such that an edge with weight smaller than which will not appear in the graph used for the
           second stage.
    :param resolution: The resolution parameter of the Louvain-like algorithm we implement in the second stage.
    :param beta_vec: The list of penalty values we use in the second stage.
    :param second_saving_path: The path in which the results of the second stage will be saved. This is only the name,
           without '.csv' and path, under which the file is saved.
    :param assess: A boolean indicating whether to print our performance (Fraction of communities with exactly one node
    of each type, and fraction of communities exactly caught).
    :param draw: A boolean indicating whether to draw the final multipartite graph, with colors indicating which
           nodes belong to which communities.
    :param community_ground_truth: A path to the file of the ground truth communities. Required if one wants to evaluate
    the performance of the model.
    """
    start = time.time()

    second_graph_filenames = []
    second_graph_ids = []
    for graph_path, graph_id, first_saving_path in zip(graph_filenames, graph_ids, first_saving_paths):
        first_saving_path_01 = first_saving_path[:-4] + "_01" + first_saving_path[-4:]
        first_saving_path_10 = first_saving_path[:-4] + "_10" + first_saving_path[-4:]
        second_graph_filenames.extend([first_saving_path_01, first_saving_path_10])
        second_graph_ids.extend([graph_id, tuple(reversed(graph_id))])
        MatchingProblem(graph_path, algorithm, first_stage_params, first_saving_path_01, row_ind=0, col_ind=1)
        MatchingProblem(graph_path, algorithm, first_stage_params, first_saving_path_10, row_ind=1, col_ind=0)

    gr = load_graph_from_files(second_graph_filenames, second_graph_ids, has_title=True, cutoff=cutoff)
    gt = load_ground_truths(community_ground_truth) if community_ground_truth is not None else None

    run_louvain(gr, second_saving_path, resolution, beta_vec, assess=assess, ground_truth=gt, draw=draw)


def run_yoram_networks():
    for network in range(2, 5):
        print(network)
        graph_files = [os.path.join("..", "BipartiteProbabilisticMatching", "data",
                                    f"Obs_Pair_K_Network_{network}_Graph_{g}.csv") for g in range(1, 4)]
        graph_node_types = [(0, i) for i in range(1, 4)]
        first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                                 f"yoram_network_{network}",
                                                 f"yoram_network_{network}_graph_{g}.csv") for g in range(1, 4)]
        # gt_file_name = os.path.join("..", "data", f"Real_Tags_K_Network_{network}.csv")
        first_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
        if not os.path.exists(os.path.join("..", "BipartiteProbabilisticMatching",
                                           "results", f"yoram_network_{network}")):
            os.mkdir(
                os.path.join("..", "BipartiteProbabilisticMatching", "results", f"yoram_network_{network}"))
        first_second_stage(graph_files, graph_node_types, "flow_numeric", first_params, first_stage_saving_paths,
                           cutoff=0.0, resolution=0., beta_vec=[10., 10., 10., 10.],
                           second_saving_path=f"yoram_network_{network}_communities", assess=False, draw=False,
                           community_ground_truth=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-T")
    parser.add_argument("--source", "-S")
    parser.add_argument("--destination", "-D")
    args = parser.parse_args()
    task_number = args.task
    source_dir = args.source
    destination_dir = args.destination
    # run_yoram_networks()
    if task_number == '1' and args.source and args.destination:
        directory_saving_name = args.destination
        rootDir = os.path.join("...", "BipartiteProbabilisticMatching", "data", source_dir)
        graph_file_names = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                            os.walk(rootDir) for file in filenames]

        first_stage_saving_paths = [os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                                 directory_saving_name,
                                                 f"yoram_network_1_graph_{g}.csv") for g in range(1, 4)]
        first_stage_params = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}

        if not os.path.exists(os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                                 directory_saving_name)):
            os.makedirs(os.path.join("..", "BipartiteProbabilisticMatching", "results",
                                     directory_saving_name))

        if len(graph_file_names) == 0:
            print("No file were found!")
        else:
            task1(graph_file_names, first_stage_saving_paths, first_stage_params)

    elif task_number == '2' and args.source and args.destination:
        second_saving_path = args.destination
        second_graph_ids = []
        for i in range(1, 4):
            second_graph_ids.append((0, i))
            second_graph_ids.append((i, 0))

        rootDir = os.path.join("..", "MultipartiteCommunityDetection", "data", source_dir)
        second_graph_filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                                  os.walk(rootDir) for file in filenames]
        gr = load_graph_from_files(second_graph_filenames, second_graph_ids, has_title=True, cutoff=0.0)

        task2(gr, second_saving_path, 0., [10., 10., 10., 10.], assess=False, ground_truth=None, draw=False)

    elif task_number == '3' and args.source:
        max_steps = 4
        starting_point = '2a'
        graph_file_names = []
        for g in range(1, 4):
            graph_file_names.append(
                os.path.join("..", "PathwayProbabilitiesCalculation", "data", source_dir, source_dir + f"_graph_{g}_01.csv"))
            graph_file_names.append(
                os.path.join("..", "PathwayProbabilitiesCalculation", "data", source_dir, source_dir + f"_graph_{g}_10.csv"))

        from_to_groups = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
        # multipartite_graph = load_graph(graph_file_names)

        print(task3(max_steps, starting_point, graph_file_names, from_to_groups))
    else:
        print("wrong/missing arguments...")
