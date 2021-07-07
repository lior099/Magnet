"""
Visualization Task for Evaluation.
"""

import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib as mpl
from state_of_the_art.state_of_the_art_embedding import *


mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 20


def our_embedding_method(final_dict_proj):
    """
    Run cheap node2vec
    :param final_dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a list of projections of every node as np arrays
    """
    keys = list(final_dict_proj.keys())
    projections = []
    for i in range(len(keys)):
        projections.append(final_dict_proj[keys[i]])
    return projections


def read_labels(file):
    """
    Read the labels file and return the labels as a list
    :param file: a file with labels for every node
    :return: a list of the labels of every node
    """
    c = np.loadtxt(file, dtype=int)
    labels = {x: y for (x, y) in c}
    list_labels = []
    keys = list(labels.keys())
    for i in range(len(keys)):
        list_labels.append(labels[keys[i]])
    return list_labels


def read_labels_only_embd_nodes(file_1, list_in_embd):
    c = np.loadtxt(file_1, dtype=int)
    labels = {str(x): y for (x, y) in c}
    list_labels = []
    for l in list_in_embd:
        list_labels.append(labels[l])
    return list_labels


def visualization(name, initial_method, method, projections, nodes, i, labels, initial):
    """
    The visualization task explained in details in the pdf file attached in the github.
    :param projections: a list of projections of every node as np arrays
    :param nodes: nodes of the graph
    :param i: number of figure
    :param labels:a list of the labels of every node
    :return: tsne representation of both regular node2vec and cheap node2vec
    """
    projections = np.asarray(projections)

    names = []
    for i in range(len(nodes)):
        names.append(nodes[i])

    data = pd.DataFrame(projections)
    X = data.values

    tsne_points = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(i, figsize=(7, 6))
    ax = fig.add_subplot(111)
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')

    plt.title('{} Visualization - {} + {} \n Size of Initial Embedding is {}'.format(name, initial_method, method, initial))

    if len(labels) == 0:
        ax.scatter(tsne_points[:, 0], tsne_points[:, 1], cmap='tab10', alpha=0.8, s=8)
    else:
        ax.scatter(tsne_points[:, 0], tsne_points[:, 1], c=labels, cmap='tab10', alpha=0.8, s=8)
    plt.savefig(os.path.join("..", "visualization_figures", "{} {} + {} + {}.png".format(name, initial_method, method, initial)))



