

import csv
from StaticGraphEmbeddings.our_embeddings_methods.static_embeddings import main_static
from StaticGraphEmbeddings.state_of_the_art.state_of_the_art_embedding import *
import networkx as nx
import math
import scipy as sp
from StaticGraphEmbeddings.evaluation_tasks.plots_utils import choose_max_initial, all_test_by_one_chosen_initial
import os


def divide_to_keys(dict_all_embeddings):
    """
    Distinguish between types of embedding methods - ours and state-of-the-art
    :param dict_all_embeddings: dict of all embeddings
    :return: Names of our methods, names of state-of-the-art methods
    """
    keys_ours = []
    keys_state_of_the_art = []
    keys = list(dict_all_embeddings.keys())
    for key in keys:
        value = dict_all_embeddings[key]
        if len(value) > 3:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)
    return keys_ours, keys_state_of_the_art


def order_by_different_methods(dict_embeddings, initial_methods):
    """
    Create from the embedding dictionary 2 necessary dicts.
    :param dict_embeddings:
    :param initial_methods:
    :return: 2 dictionaries:
                1. dict_of_connections: key == state-of-the-art-method , value == list of our methods that key is their
                    initial embedding
                2. dict_methods: key == state-of-the-art-method , value == list of embedding dictionaries of the methods
                    that key is their initial embedding
    """
    dict_connections_methods = {}
    dict_of_methods = {}
    for m in initial_methods:
        dict_connections_methods.update({m: []})
        dict_of_methods.update({m: []})
    keys = list(dict_embeddings.keys())
    for key in keys:
        my_list = dict_embeddings[key]
        if len(my_list) > 3:
            our_method = my_list[0]
            method = my_list[1]
            dict_connections_methods[method].append(our_method)
            dict_of_methods[method].append(dict_embeddings[key])
        else:
            dict_of_methods[key].append(dict_embeddings[key])
    return dict_of_methods, dict_connections_methods


def order_results_by_different_initial_methods(dict_embeddings, initial_methods, dict_mission):
    """
    Create from the mission dictionary (containing results for each embeddings method) 1 necessary dict.
    :param dict_embeddings: Dictionary of all embeddings
    :param initial_methods: List of names of state-of-the-art embedding methods
    :param dict_mission: Dictionary of the mission, containing results (scores)
    :return: dict_of_methods: key == state-of-the-art-method , value == list of mission dictionaries of the methods
                    that key is their initial embedding, including the initial embedding itself.
    """
    keys = list(dict_embeddings.keys())
    dict_of_methods = {}
    for m in initial_methods:
        dict_of_methods.update({m: []})
    for key in keys:
        my_list = dict_embeddings[key]
        if len(my_list) > 3:
            method = my_list[1]
            dict_of_methods[method].append(dict_mission[key])
        else:
            dict_of_methods[key].append(dict_mission[key])
    return dict_of_methods


def calculate_std(r, i, dict_mission, keys, keys_ours, keys_state_of_the_art):
    """
    Calculate the variance of the results
    """
    all_micro = []
    all_macro = []
    all_auc = []
    for key in keys_ours:
        dict_initial = dict_mission[key]
        micro_f1 = dict_initial[r][0][i]
        macro_f1 = dict_initial[r][1][i]
        auc = dict_initial[r][3][i]
        all_micro.append(micro_f1)
        all_macro.append(macro_f1)
        all_auc.append(auc)
    for key in keys_state_of_the_art:
        dict_initial = dict_mission[key]
        micro_f1 = dict_initial[r][0][0]
        macro_f1 = dict_initial[r][1][0]
        auc = dict_initial[r][3][0]
        all_micro.append(micro_f1)
        all_macro.append(macro_f1)
        all_auc.append(auc)
    std_micro = str(round(np.std(all_micro), 3))
    std_macro = str(round(np.std(all_macro), 3))
    std_auc = str(round(np.std(all_auc), 3))
    return std_micro, std_macro, std_auc


def create_dicts_for_results(dict_all_embeddings, dict_mission, our_initial, n):
    """
    Create dictionary of results and more information to create useful csv files of results for a given mission
    :param dict_all_embeddings: Dictionary f embeddings
    :param dict_mission: Dictionary of the given mission
    :param our_initial: Array of different sizes of initial embedding.
    :param n: Number of nodes in the graph.
    :return:
    """
    keys_ours, keys_state_of_the_art = divide_to_keys(dict_all_embeddings)
    keys = list(dict_all_embeddings.keys())

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = dict_all_embeddings[key][1]
            regression = dict_all_embeddings[key][0]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
            t = round(dict_all_embeddings[key][2], 3)
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys_ours, keys_state_of_the_art)
                if key in keys_ours:
                    t = round(dict_all_embeddings[key][8][i])
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1)+"+-"+std_micro,
                                "macro-f1": str(macro_f1)+"+-"+std_macro, "auc": str(auc)+"+-"+std_auc, "time": t}
                list_dicts.append(dict_results)
    return list_dicts


def export_results(n, dict_all_embeddings, dict_mission, our_initial, name, mission):
    """
    Write the results to csv file
    """
    csv_columns = ["initial size", "embed algo", "regression", "test", "micro-f1", "macro-f1", "auc", "time"]
    dict_data = create_dicts_for_results(dict_all_embeddings, dict_mission, our_initial, n)
    csv_file = os.path.join("..", "files", "{} {}.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def export_results_from_files(dict_all_embeddings, dict_mission, our_initial, n):
    """
    Export results from Link Prediction / Node Classification Results files
    """
    keys = list(dict_all_embeddings.keys())
    keys_ours = []
    keys_state_of_the_art = []
    for key in keys:
        if "+" in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = key.split(" + ")[0]
            regression = key.split(" + ")[1]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys_ours,
                                                              keys_state_of_the_art)
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1) + "+-" + std_micro,
                                "macro-f1": str(macro_f1) + "+-" + std_macro, "auc": str(auc) + "+-" + std_auc}
                list_dicts.append(dict_results)
    return list_dicts


"""
Export tasks without time
"""


def export_results_lp_nc(dict_all_embeddings, dict_mission, our_initial, n):
    keys = list(dict_all_embeddings.keys())
    keys_ours = []
    keys_state_of_the_art = []
    for key in keys:
        if "+" in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = key.split(" + ")[0]
            regression = key.split(" + ")[1]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys, keys_ours,
                                                              keys_state_of_the_art)
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1) + "+-" + std_micro,
                                "macro-f1": str(macro_f1) + "+-" + std_macro, "auc": str(auc) + "+-" + std_auc}
                list_dicts.append(dict_results)
    return list_dicts


def export_results_lp_nc_all(n, save, dict_all_embeddings, dict_mission, our_initial, name, mission):
    csv_columns = ["initial size", "embed algo", "regression", "test", "micro-f1", "macro-f1", "auc"]
    dict_data = export_results_lp_nc(dict_all_embeddings, dict_mission, our_initial, n)
    csv_file = os.path.join("..", save, "{} {}.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def export_best_results(name, mission, dict_mission, keys_ours_, keys_state_of_the_art_ ,ratio_arr, initial_arr, scores, index_ratio):
    csv_columns = ["embed algo", "regression", "micro-f1", "macro-f1", "auc"]
    list_dicts = []
    for key in keys_ours_:
        all_scores = []
        for score in scores:
            dict_number_initial = choose_max_initial(dict_mission, keys_ours_, ratio_arr, initial_arr, score)
            dict_test_score = all_test_by_one_chosen_initial(dict_mission, dict_number_initial, keys_state_of_the_art_,
                                                             ratio_arr, score)
            my_score = dict_test_score[key][index_ratio]
            all_scores.append(my_score)
        dict_results = {"embed algo": key.split(" + ")[0] , "regression": key.split(" + ")[1],
                        "micro-f1": str(all_scores[0]),
                        "macro-f1": str(all_scores[1]), "auc": str(all_scores[2])}
        list_dicts.append(dict_results)
    csv_file = os.path.join("..", "files", "{} {} best results.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in list_dicts:
                writer.writerow(data)
    except IOError:
        print("I/O error")



