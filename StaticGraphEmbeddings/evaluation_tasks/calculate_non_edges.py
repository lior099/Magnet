"""
For link prediction mission, one needs a csv file of non edges of each graph. This file creates them.
For your own dataset, you need its name and path to where it can be found (see downward). If your dataset needs a
different way of loading, add it. In the end you need a networkx graph.
"""

from eval_utils import load_data
import os
import itertools as IT
import csv
import json
import random
import networkx as nx

name1 = "DBLP"
name2 = "Cora"
name3 = "Pubmed"
name4 = "Yelp"
name5 = "Reddit"

names = [name1, name2, name3, name4, name5]

datasets_path = os.path.join("..", "datasets")
labels_path = os.path.join("..", "labels")

# how many false edges do you want to extract? try as large as possible.
number = 20000

for name in names:
    if name == "Pubmed":
        G = nx.read_edgelist(os.path.join(datasets_path, name + ".txt"), delimiter=',')
    else:
        if name == "Yelp":
            H = load_data("yelp.npz")
            mapping = {}
            nodes = list(H.nodes())
            with open(os.path.join(labels_path, "class_map.json")) as json_file:
                data = json.load(json_file)
            keys = list(data.keys())
            for i in range(len(keys)):
                mapping.update({nodes[i]: keys[i]})
            G = nx.relabel_nodes(H, mapping)
        else:
            G = nx.read_edgelist(os.path.join(datasets_path, name + ".txt"))

    list_in_embd = list(G.nodes())

    # choosing randomly indices of the nodes because non-edge is too big for all nodes
    indexes = random.sample(range(1, len(list_in_embd)), number)
    new_list = []
    for l in indexes:
        new_list.append(list_in_embd[l])
    sub_G = G.subgraph(new_list)
    print(sub_G.number_of_nodes())
    # create a list of all missing edges of the nodes that were chosen randomly
    missing = [pair for pair in IT.combinations(sub_G.nodes(), 2) if not sub_G.has_edge(*pair)]
    print(len(missing))
    print(1)

    # extract list to a csv file
    csvfile = open('non_edges_{}.csv'.format(name), 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerows(missing)
    csvfile.close()

