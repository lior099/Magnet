"""
Create the plots, only when there are files of link prediction and node classification
"""

from plots_utils import *

# dictionary of important parameter for your dataset. Here are some examples
DATASET = {"name": "Cora", "label_file": "cora_tags.txt", "initial_embedding_size": [], "dim": 128}
# DATASET = {"name": "DBLP", "label_file": "dblp_tags.txt", "initial_embedding_size": [], "dim": 128}
# DATASET = {"name": "Cora", "label_file": "pubmed2_tags.txt", "initial_embedding_size": [], "dim": 128}

# paths to where the dataset and label file are in
datasets_path = os.path.join("..", "datasets")
labels_path = os.path.join("..", "labels")

# the graph and number of nodes
if DATASET["name"] == "Pubmed":
    G = nx.read_edgelist(os.path.join(datasets_path, DATASET["name"] + ".txt"), create_using=nx.DiGraph(), delimiter=",")
elif DATASET["name"] == "Yelp":
    G = load_data("amazon.npz")
else:
    G = nx.read_edgelist(os.path.join(datasets_path, DATASET["name"] + ".txt"), create_using=nx.DiGraph())
n = G.number_of_nodes()

list_keys = []
methods_ = ["OGRE", "DOGRE", "WOGRE"]
initial_methods_ = ["node2vec", "HOPE", "GF"]
methods_mapping = {"VERSE": "OGRE", "D-VERSE": "DOGRE", "We-VERSE": "WOGRE", "LGF": "LGF"}
# methods_mapping = {"OGRE": "OGRE", "DOGRE": "DOGRE", "WOGRE": "WOGRE", "LGF": "LGF"}
for i in initial_methods_:
    for m in methods_:
        list_keys.append(i + " + " + m)
if DATASET["name"] != "Yelp":
    if DATASET["name"] != "Reddit":
        list_keys.remove("HOPE + LGF")
        list_keys.remove("node2vec + LGF")
keys_ours = list_keys
keys_state_of_the_art = initial_methods_
keys = keys_ours + keys_state_of_the_art

# for the plots
num1 = 170
num2 = 180
num3 = 10
num4 = 30

# where to read the files from
save1 = "files_degrees"
# where to save the plot
save2 = "plots"

"""
Plot Running Time
"""

# read times
dict_times = read_times_file(DATASET["name"], "files", initial_methods_, methods_mapping)

# colors to plot each method in
colors = ["indigo", "red", "olivedrab"]

mapping = {}
all_keys = keys_ours + keys_state_of_the_art
for key in all_keys:
    if "node2vec" in key:
        mapping.update({key: colors[0]})
    elif "HOPE" in key:
        mapping.update({key: colors[1]})
    else:
        mapping.update({key: colors[2]})

plot_running_time_after_run_all(DATASET, dict_times, mapping, n, 0)

"""
Link Prediction and Node Classification
"""
dict_lp, _, initial_size, _, _ = read_results(DATASET["name"], save1, "Link Prediction", initial_methods_, 0.2, methods_mapping)
dict_nc = read_results(DATASET["name"], save1, "Node Classification", initial_methods_, 0.2, methods_mapping)

# export_best_results(DATASET["name"], "Node Classification", dict_nc, keys_ours, keys_state_of_the_art, [0.1, 0.2, 0.4, 0.7], initial_size,
#                             ["Micro-F1", "Macro-F1", "AUC"], 1)
# export_best_results(DATASET["name"], "Link Prediction", dict_nc, keys_ours, keys_state_of_the_art, [0.1, 0.2, 0.4, 0.7], initial_size,
#                             ["Micro-F1", "Macro-F1", "AUC"], 1)

new_initial = []
for i in initial_size:
    new_initial.append(i)
new_initial.append(n)

params_dict_ = [{"dimension": 128, "walk_length": 80, "num_walks": 16, "workers": 2}, {"dimension": 128, "beta": 0.1},
               {"dimension": 128, "eta": 0.1, "regularization": 0.1, "max_iter": 8000, "print_step": 50}]

DATASET["initial_embedding_size"] = initial_size
ONE_TEST_RATIO = 0.2

# lp and nc parameters dictionary
params_lp_dict = {"number_true_false": 10000, "rounds": 10, "test_ratio": [0.1, 0.2, 0.4, 0.7], "number_of_sub_graphs": 10}
params_nc_dict = {"rounds": 10, "test_ratio": [0.1, 0.2, 0.4, 0.7]}

plot_test_vs_score_all(DATASET["name"], "Link Prediction", dict_lp, keys_ours, keys_state_of_the_art, params_lp_dict["test_ratio"],
                       DATASET["initial_embedding_size"], mapping, num3, save2)
plot_test_vs_score_all(DATASET["name"], "Node Classification", dict_nc, keys_ours, keys_state_of_the_art,
                       params_nc_dict["test_ratio"],
                       DATASET["initial_embedding_size"], mapping, num4, save2)

plot_initial_vs_score_all(DATASET["name"], "Link Prediction", dict_lp, keys_ours, keys_state_of_the_art,
                       DATASET["initial_embedding_size"], ONE_TEST_RATIO, params_lp_dict["test_ratio"], new_initial, mapping, num1, save2)
plot_initial_vs_score_all(DATASET["name"], "Node Classification", dict_nc, keys_ours, keys_state_of_the_art,
                       DATASET["initial_embedding_size"], ONE_TEST_RATIO, params_nc_dict["test_ratio"], new_initial, mapping, num2, save2)
