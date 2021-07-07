

import sys
import os

from StaticGraphEmbeddings.state_of_the_art.state_of_the_art_embedding import save_embedding_state_of_the_art, final

for root, dirs, files in os.walk('StaticGraphEmbeddings'):
    sys.path.append(os.path.abspath(root))
    for dir in dirs:
        sys.path.append(os.path.abspath(os.path.join(root, dir)))
from StaticGraphEmbeddings.evaluation_tasks.link_prediction import *
from StaticGraphEmbeddings.evaluation_tasks.node_classification import *
from StaticGraphEmbeddings.our_embeddings_methods.static_embeddings import *
import csv

def load_graph(path, name, is_weighted):
    """
    Data loader assuming the format is a text file with columns of : target source (e.g. 1 2) or target source weight
    (e.g. 1 2 0.34). If you have a different format, you may want to create your own data loader.
    :param path: The path to the edgelist file
    :param name: The name of te dataset
    :param is_weighted: True if the graph is weighted, False otherwise.
    :return: A Directed networkx graph with an attribute of "weight" for each edge.
    """
    if name == "Yelp":
        with open(os.path.join(path, "yelp_data.p"), 'rb') as f:
            G = pickle.load(f)
        G = add_weights(G)
    else:
        if is_weighted is True:
            if name == "Pubmed":
                G = nx.read_weighted_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph(),
                                              delimiter=",")
            else:
                # where the file is in the format : source target weight
                G = nx.read_weighted_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph())
        else:
            if name == "Pubmed":
                G = nx.read_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph(), delimiter=",")
            else:
                # where the file is in the format : source target , so we put weight=1 for each edge
                G = nx.read_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph())
            G = add_weights(G)
    return G


def calculate_static_embeddings(datasets_path, embeddings_path, dict_dataset, methods, initial_methods, params_dict,
                                from_files=False, save_=False, only_initial=False):
    """
    Function to calculate static embedding, both by ours and state-of-the-art methods.
    :param datasets_path: Path to where the datasets are
    :param embeddings_path: Path to were the embeddings meed to be saved. Notice state-of-the-art methods are saved in
                            a different path.
    :param dict_dataset: Dict of dataset's important parameters, see example later.
    :param methods: List of state-of-the-art methods for initial embedding- "node2vec", "HOPE" or "GF"/
    :param initial_methods: List of our suggested embedding methods- "OGRE", "DOGRE", "WOGRE" or "LGF".
    :param params_dict: Dict of parameters for state-of-the-art methods
    :param from_files: False if you want to calculate the embeddings, True if you want to read them from a .npy file
                       format.
    :param save_: True ig you want to save the embeddings, else False.
    :return: A dictionary where keys are embedding methods that were applied and values are list of embedding dicts
             for each embedding method.
    """
    name = dict_dataset["name"]
    initial_size = dict_dataset["initial_size"]
    dim = dict_dataset["dim"]
    is_weighted = dict_dataset["is_weighted"]
    choose = dict_dataset["choose"]
    regu_val = dict_dataset["regu_val"]
    weighted_reg = dict_dataset["weighted_reg"]
    s_a = dict_dataset["s_a"]
    epsilon = dict_dataset["epsilon"]
    file_tags = dict_dataset["label_file"]

    from MultipartiteCommunityDetection.code.run_louvain import load_graph_from_files
    G = load_graph_from_files(datasets_path, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)], has_title=True, cutoff=0.0)

    if from_files is False:

        my_dict = {}
        state_of_the_art_dict = {}

        if not only_initial:
            for i in range(len(methods)):
                for j in range(len(initial_methods)):
                    if methods[i] == "LGF" and initial_methods[j] != "GF":
                        continue
                    # print("start {} + {}".format(methods[i], initial_methods[j]))
                    # calculate static embedding by given method
                    SE = StaticEmbeddings(name, G, initial_size, initial_method=initial_methods[j], method=methods[i],
                                          dim=dim, choose=choose, regu_val=regu_val, weighted_reg=weighted_reg,
                                          epsilon=epsilon, file_tags=file_tags)
                    if save_:
                        SE.save_embedding(embeddings_path)
                    key = "{} + {}".format(methods[i], initial_methods[j])
                    # save the embedding in a dictionary with all embedding methods
                    my_dict.update({key: SE})
                    if i == 0:
                        mmm = SE.initial_size
                        list_initial_nodes = SE.list_initial_proj_nodes

        if s_a:
            if name != "Yelp":
                if name != "Reddit":
                    for im in initial_methods:
                        X, projections, t = final(G, im, params_dict[im])
                        state_of_the_art_dict.update({im: [X, projections, t]})
                        if save_:
                            save_embedding_state_of_the_art(os.path.join("..", "embeddings_state_of_the_art"),
                                                            projections, name, im)

        z = {**my_dict, **state_of_the_art_dict}
        # print(list(z.keys()))

    else:
        z = {}
        for im in initial_methods:
            dict_embedding = load_embedding(os.path.join("..", "embeddings_state_of_the_art"), name + " + " + im)
            z.update({im: dict_embedding})
            for m in methods:
                for l in initial_size:
                    dict_embedding = load_embedding(embeddings_path, name + " + " + im + " + " + m + " + " + str(l))
                    z.update({f"{im} + {m}": dict_embedding})

    return z, G, 0, 0


def export_time(z, name):
    """
    Export running times to csv file.
    :param z: Dict of lists of embeddings dicts.
    :param name: Name of the dataset
    """
    list_dicts = []
    csv_columns = ["initial size", "embed algo", "regression", "time"]
    csv_file = os.path.join("..", "files", "{} times_class.csv".format(name))
    keys = list(z.keys())
    for key in keys:
        if " + " in key:
            se = z[key]
            initial_method = se.initial_method
            method = se.embedding_method
            for j in range(len(se.list_dicts_embedding)):
                data_results = {"initial size": se.initial_size[j], "embed algo": initial_method, "regression": method,
                                "time": se.times[j]}
                list_dicts.append(data_results)
        else:
            data_results = {"initial size": "", "embed algo": key, "regression": "", "time": z[key][2]}
            list_dicts.append(data_results)
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in list_dicts:
            writer.writerow(data)

def ogre_static_embeddings(graph_file_names, epsilon):
    DATASET = {"name": "Test", "initial_size": 100, "dim": 128, "is_weighted": True, "choose": "degrees",
               "regu_val": 0, "weighted_reg": False, "s_a": True, "epsilon": epsilon,
               "label_file": os.path.join("..", "labels", "cora_tags.txt")}
    # datasets_path_ = os.path.join("..", "datasets")
    # where to save the embeddings
    if DATASET["choose"] == "degrees":
        embeddings_path_ = os.path.join("..", "embeddings_degrees")
    else:
        embeddings_path_ = os.path.join("..", "embeddings_k_core")
    # Our suggested embedding method
    methods_ = ["OGRE"]
    # state-of-the-art embedding methods
    initial_methods_ = ["node2vec"]
    # Parameters duct for state-of-the-art embedding methods
    params_dict_ = {"node2vec": {"dimension": DATASET["dim"], "walk_length": 80, "num_walks": 16, "workers": 2},
                    "GF": {"dimension": DATASET["dim"], "eta": 0.1, "regularization": 0.1, "max_iter": 3000,
                           "print_step": 100}, "HOPE": {"dimension": 128, "beta": 0.1},
                    "GCN": {"dimension": DATASET["dim"], "epochs": 150, "lr": 0.01, "weight_decay": 5e-4, "hidden": 200,
                            "dropout": 0}}
    # if you want to read embedding from npy files, from_files=True. Norice they must be in an npy format and consistent
    # to the initialized parameters above.
    from_files = False
    # if you want to save the embeddings as npy file- save_=True
    save_ = False

    # calculate dict of embeddings
    return calculate_static_embeddings(graph_file_names, embeddings_path_, DATASET,
                                                                              methods_, initial_methods_, params_dict_,
                                                                              from_files=from_files, save_=save_)


def main():
    # initialize important variables / parameters
    DATASET = {"name": "Cora", "initial_size": [100, 500], "dim": 128, "is_weighted": False, "choose": "degrees",
               "regu_val": 0, "weighted_reg": False, "s_a": True, "epsilon": 0.1, "label_file": os.path.join("..", "labels", "cora_tags.txt")}
    datasets_path_ = os.path.join("..", "datasets")
    # where to save the embeddings
    if DATASET["choose"] == "degrees":
        embeddings_path_ = os.path.join("..", "embeddings_degrees")
    else:
        embeddings_path_ = os.path.join("..", "embeddings_k_core")
    # Our suggested embedding method
    methods_ = ["OGRE"]
    # state-of-the-art embedding methods
    initial_methods_ = ["node2vec"]
    # Parameters duct for state-of-the-art embedding methods
    params_dict_ = {"node2vec": {"dimension": DATASET["dim"], "walk_length": 80, "num_walks": 16, "workers": 2},
                    "GF": {"dimension": DATASET["dim"], "eta": 0.1, "regularization": 0.1, "max_iter": 3000,
                           "print_step": 100}, "HOPE": {"dimension": 128, "beta": 0.1},
                    "GCN": {"dimension": DATASET["dim"], "epochs": 150, "lr": 0.01, "weight_decay": 5e-4, "hidden": 200,
                            "dropout": 0}}
    # if you want to read embedding from npy files, from_files=True. Norice they must be in an npy format and consistent
    # to the initialized parameters above.
    from_files = False
    # if you want to save the embeddings as npy file- save_=True
    save_ = True

    # calculate dict of embeddings
    z, G, initial_size, list_initial_proj_nodes = calculate_static_embeddings(datasets_path_, embeddings_path_, DATASET,
                                                                              methods_, initial_methods_, params_dict_,
                                                                              from_files=from_files, save_=save_)

    """
    if the embeddings is all you wanted you can stop here. Otherwise, here are fnctions to calculate running time, and 
    applying Link Prediction and Node Classification Tasks.
    """

    # evaluate running time
    # export_time(z, DATASET["name"])

    # where to save resuts files
    if DATASET["choose"] == "degrees":
        save = "files_degrees"
    else:
        save = "files_k_core"

    if DATASET["name"] == "Yelp":
        mapping = {i: n for i,n in zip(range(G.number_of_nodes()), list(G.nodes()))}
    else:
         mapping=None

    DATASET["initial_size"] = initial_size
    # print(initial_size)

    # Link prediction Task
    n = G.number_of_nodes()
    non_edges_file = "non_edges_{}.csv".format(DATASET["name"])
    params_lp_dict = {"number_true_false": 1000, "rounds": 10, "test_ratio": [0.1, 0.2], "number_choose": 10}
    dict_lp = final_link_prediction(z, params_lp_dict, non_edges_file)
    export_results_lp_nc_all(n, save, z, dict_lp, DATASET["initial_size"], DATASET["name"], "Link Prediction")
    print("finish link prediction")

    # Node Classification Task
    params_nc_dict = {"rounds": 10, "test_ratio": [0.1, 0.2]}
    dict_nc = final_node_classification(DATASET["name"], z, params_nc_dict, DATASET, mapping=mapping)
    export_results_lp_nc_all(n, save, z, dict_nc, DATASET["initial_size"], DATASET["name"], "Node Classification")
    print("finish node classification")

if __name__ == "__main__":
    print("TEST")

