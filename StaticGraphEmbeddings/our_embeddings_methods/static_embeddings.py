"""
Our suggested static embedding methods full calculation.
"""
import math
from StaticGraphEmbeddings.our_embeddings_methods.utils import *
from StaticGraphEmbeddings.state_of_the_art.state_of_the_art_embedding import *
from StaticGraphEmbeddings.our_embeddings_methods.LGF import *
from StaticGraphEmbeddings.our_embeddings_methods.OGRE import *
from StaticGraphEmbeddings.our_embeddings_methods.D_W_OGRE import main_D_W_OGRE
import time


class StaticEmbeddings:
    """
    Class to run one of our suggested static embedding methods.
    """
    def __init__(self, name, G, initial_size=100, initial_method="node2vec", method="OGRE", H=None,
                 dim=128, choose="degrees", regu_val=0, weighted_reg=False, epsilon=0.1, file_tags=None):
        """
        Init function to initialize the class
        :param name: Name of the graph/dataset
        :param G: Our graph
        :param initial_method: Initial state-of-the-art embedding algorithm for the initial embedding. Options are
                "node2vec" , "gf", "HOPE" or "GCN". Default is "node2vec".
        :param method: One of our suggested static embedding methods. Options are "OGRE", "DOGRE" or "WOGRE". Default
                is "OGRE".
        :param initial_size: Size of initial embedding (integer that is less or equal to the number of nodes in the
                graph). Default value is 100.
        :param H: a networkx graph - If you already have an existing sub graph for the initial embedding, insert it as input as networkx
                  graph (initial_size is not needed in this case), else None.
        :param dim: Embedding dimension. Default is 128.
        :param choose: Weather to choose the nodes of the initial embedding by highest degree or highest k-core score.
                Options are "degrees" for the first and "k-core" for the second.
        :param regu_val: If DOGRE/WOGRE method is applied, one can have a regression with regularization, this is the value
                of the regularization coefficient. Default is 0 (no regularization).
        :param weighted_reg: If DOGRE/WOGRE method is applied, one can have a weighted regression. True for weighted
                regression, else False. Default is False.
        :param epsilon: The weight that is given to the embeddings of the second order neighbours in OGRE (no need in DOGRE/WOGRE).
        :param file tags: If the initial embedding is GCN, file tags is needed. You can see an example format in "labels" directory.
        """
        self.name = name
        # The graph which needs to be embed. If you have a different format change to your own loader.
        self.graph = G
        self.initial_method = initial_method
        self.embedding_method = method
        if H is None:
            if type(initial_size) == int:
                self.initial_size = [initial_size]
            elif len(initial_size) >= 1:
                self.initial_size = initial_size
            else:
                self.initial_size = calculate_factor_initial_size(self.graph.number_of_nodes(), math.sqrt(10))
        else:
            self.initial_size = [H.number_of_nodes()]
        # print("initial size: ", self.initial_size)
        self.dim = dim
        # dictionary of parameters for state-of-the-art method
        self.params_dict = self.define_params_for_initial_method()
        self.choose = choose
        # calculate the given graph embedding and return a dictionary of nodes as keys and embedding vectors as values,
        self.list_dicts_embedding, self.times, self.list_initial_proj_nodes = self.calculate_embedding(regu_val, weighted_reg, epsilon, file_tags, H)
        
    def define_params_for_initial_method(self):
        """
        According to the initial state-of-the-art embedding method, create the dictionary of parameters.
        :return: Parameters dictionary
        """
        if self.initial_method == "node2vec":
            params_dict = {"dimension": self.dim, "walk_length": 80, "num_walks": 16, "workers": 2}
        elif self.initial_method == "gf":
            params_dict ={"dimension": self.dim, "eta": 0.1, "regularization": 0.1, "max_iter": 3000, "print_step": 100}
        elif self.initial_method == "HOPE":
            params_dict = {"dimension": self.dim, "beta": 0.1}
        elif self.initial_method == "GCN":
            params_dict = {"dimension": self.dim, "epochs": 100, "lr": 0.1, "weight_decay": 0, "hidden": 2000,
                           "dropout": 0.2}
        else:
            params_dict = None
        return params_dict

    def calculate_embedding(self, regu_val, weighted_reg, epsilon, file_tags=None, H=None):
        """
        Calculate the graph embedding.
        :return: An embedding dictionary where keys are the nodes that are in the final embedding and values are
                their embedding vectors.
        """
        list_dicts, times, list_initial_proj_nodes = main_static(self.embedding_method, self.initial_method, self.graph, self.initial_size,
                                        self.dim, self.params_dict, self.choose,
                                        regu_val=regu_val, weighted_reg=weighted_reg, epsilon=epsilon, file_tags=file_tags, F=H)
        return list_dicts, times, list_initial_proj_nodes

    def save_embedding(self, path):
        """
        Save the calculated embedding in a .npy file.
        :param path: Path to where to save the embedding
        :return: The file name
        """
        for j in range(len(self.list_dicts_embedding)):
            dict_embedding = self.list_dicts_embedding[j]
            file_name = self.name + " + " + self.initial_method + " + " + self.embedding_method + " + " \
                        + str(self.initial_size[j])
            np.save(os.path.join(path, '{}.npy'.format(file_name)), dict_embedding)


def add_weights(G):
    """
    If the graph is not weighted, add weights equal to 1.
    :param G: The graph
    :return: The weighted version of the graph
    """
    edges = list(G.edges())
    for e in edges:
        G[e[0]][e[1]] = {"weight": 1}
    return G


def calculate_factor_initial_size(n, key):
    """
    Calculate different initial embedding sizes by a chosen factor.
    :param n: Number of nodes in the graph
    :param key: The factor- for example if key==10, the sizes will be n/10, n/100, n/100, .... the minimum is 100 nodes
    in the initial embedding
    :return: List of initial embedding sizes
    """
    initial = []
    i = n
    while i > 100:
        i = int(i/key)
        i_2 = int(i/2)
        key = pow(key, 2)
        if i > 100:
            initial.append(i)
        if i_2 > 100:
            initial.append(i_2)
    initial.append(100)
    ten_per = int(n/10)
    initial.append(ten_per)
    initial.sort()
    return initial


def load_embedding(path, file_name):
    """
    Given a .npy file - embedding of a given graph. return the embedding dictionary
    :param path: Where this file is saved.
    :param file_name: The name of the file
    :return: Embedding dictionary
    """
    data = np.load(os.path.join(path, '{}.npy'.format(file_name)), allow_pickle=True)
    dict_embedding = data.item()
    return dict_embedding

def add_weights(G):
    edges = list(G.edges())
    for e in edges:
        G[e[0]][e[1]] = {"weight": 1}
    return G

def main_static(method, initial_method, G, initial, dim, params, choose="degrees", regu_val=0., weighted_reg=False,
                epsilon=0.1, file_tags=None, F=None):
    """
    Main function to run our different static embedding methods- OGRE, DOGRE, WOGRE, LGF.
    :param method: One of our methods - OGRE, DOGRE, WOGRE, LGF (string)
    :param initial_method: state-of-the-art algorithm for initial embedding - node2vec, HOPE, GF or GCN (string)
    :param G: The graph to embed (networkx graph)
    :param initial: A list of different sizes of initial embedding (in any length, sizes must be integers).
    :param dim: Embedding dimension (int)
    :param params: Dictionary of parameters for the initial algorithm
    :param choose: How to choose the nodes in the initial embedding - if == "degrees" nodes with highest degrees are
    chosen, if == "k-core" nodes with highest k-core score are chosen.
    :param regu_val: If DOGRE/WOGRE method is applied, one can have a regression with regularization, this is the value
    of the regularization coefficient. Default is 0 (no regularization).
    :param weighted_reg: If DOGRE/WOGRE method is applied, one can have a weighted regression. True for weighted regression,
    else False. Default is False.
    :param epsilon: Determine the weight given to the second order neighbours (only in OGRE method).
    :param file tags: If the initial embedding is GCN, file tags is needed. You can see an example format in "labels" directory.
    :param F: a networkx graph - If you already have an existing sub graph for the initial embedding, insert it as input as networkx
                  graph (initial_size is not needed in this case), else None.
    :return: - list of embedding dictionaries, each connected to a different initial embedding size. The keys of the
             dictionary are the nodes that in the final embedding, values are the embedding vectors.
             - list of times - each member is the running time of the embedding method, corresponding to the matching
             size of initial embedding.
             - list of nodes that are in the initial embedding.

    """
    if method == "DOGRE":
        list_dicts, times, list_initial_proj_nodes = main_D_W_OGRE(G, initial_method, method, initial, dim, params, choose, regu_val,
                                                   weighted_reg, file_tags=file_tags, F=F)
    elif method == "WOGRE":
        list_dicts, times, list_initial_proj_nodes = main_D_W_OGRE(G, initial_method, method, initial, dim, params, choose, regu_val,
                                                   weighted_reg, file_tags=file_tags, F=F)

    else:
        user_wish = True

        # choose number of nodes in initial projection. These value corresponds to 116 nodes
        list_dicts = []
        list_initial_proj_nodes = []

        times = []
        for l in initial:
            t = time.time()
            # get the initial projection by set and list to help us later
            if F is not None:
                initial_proj_nodes = list(F.nodes())
            else:
                if choose == "degrees":
                    initial_proj_nodes = get_initial_proj_nodes_by_degrees(G, l)
                else:
                    initial_proj_nodes = get_initial_proj_nodes_by_k_core(G, l)
            list_initial_proj_nodes.append(initial_proj_nodes)
            user_print("number of nodes in initial projection is: " + str(len(initial_proj_nodes)), user_wish)
            n = G.number_of_nodes()
            e = G.number_of_edges()
            user_print("number of nodes in graph is: " + str(n), user_wish)
            user_print("number of edges in graph is: " + str(e), user_wish)
            # the nodes of our graph
            G_nodes = list(G.nodes())
            set_G_nodes = set(G_nodes)
            set_proj_nodes = set(initial_proj_nodes)
            # convert the graph to undirected
            H = G.to_undirected()
            # calculate neighbours dictionary
            neighbors_dict = create_dict_neighbors(H)
            # making all lists to set (to help us later in the code)
            set_nodes_no_proj = set_G_nodes - set_proj_nodes
            # create dicts of connections
            dict_node_node, dict_node_enode, dict_enode_enode = create_dicts_of_connections(set_proj_nodes,
                                                                                            set_nodes_no_proj,
                                                                                            neighbors_dict)
            # creating sub_G to do node2vec on it later
            if F is None:
                sub_G = create_sub_G(initial_proj_nodes, G)
            else:
                sub_G = F.copy()
            user_print("calculate the projection of the sub graph with {}...".format(initial_method), user_wish)
            if initial_method == "GF":
                my_iter = params["max_iter"]
                params["max_iter"] = 1500
                _, dict_projections, _ = final(sub_G, initial_method, params)
                params["max_iter"] = my_iter
            elif initial_method == "GCN":
                _, dict_projections, _ = final(sub_G, initial_method, params, file_tags)
            else:
                _, dict_projections, _ = final(sub_G, initial_method, params)
            if method == "LGF":
                final_dict_enode_proj, set_n_e = final_function_LGF(dict_projections, dict_node_enode, dict_node_node,
                                                                   dict_enode_enode, set_nodes_no_proj, 0.01, dim)
            elif method == "OGRE":
                final_dict_enode_proj, set_n_e = final_function_OGRE(dict_projections, dict_node_enode,
                                                    dict_node_node, dict_enode_enode, set_nodes_no_proj, 0.01, dim,
                                                                     H, epsilon=epsilon)
            else:
                print("non-valid embedding method")
                break

            print("The number of nodes that aren't in the final projection:", len(set_n_e))
            user_print("calculate remaining nodes embedding with {}".format(initial_method), user_wish)
            if len(set_n_e) != 0:
                set_n_e_sub_g = nx.subgraph(G, list(set_n_e))
                if initial_method == "GCN":
                    _, projections, _ = final(set_n_e_sub_g, initial_method, params, file_tags=file_tags)
                elif initial_method == "HOPE":
                    if len(set_n_e) < int(dim/2):
                        params = {"dimension": dim, "walk_length": 80, "num_walks": 16, "workers": 2}
                        _, projections, _ = final(set_n_e_sub_g, "node2vec", params)
                    else:
                        _, projections, _ = final(set_n_e_sub_g, initial_method, params)
                else:
                    _, projections, _ = final(set_n_e_sub_g, initial_method, params)
            else:
                projections = {}
            z = {**final_dict_enode_proj, **projections}
            print("The number of nodes that are in the final projection:", len(z))
            elapsed_time = time.time() - t
            times.append(elapsed_time)
            print("running time: ", elapsed_time)

            list_dicts.append(z)
            
    return list_dicts, times, list_initial_proj_nodes


"""
Example code to calculate embedding can be seen in the file- evaluation_tasks/calculate_static_embeddings.py.
"""

# name = "Cora"  # name
# file_tags = "../labels/cora_tags.txt"     # labels file
# dataset_path = os.path.join("..", "datasets")     # path to where datasets are saved
# embeddings_path = os.path.join("..", "embeddings_degrees")     # path to where embeddings are saved
# G = nx.read_edgelist(os.path.join(dataset_path, name + ".txt"), create_using=nx.DiGraph())    # read the graph
# G = add_weights(G)   # add weights 1 to the graph if it is unweighted
# initial_method = "node2vec"     # known embedding algorithm to embed initial vertices with
# method = "OGRE"        # our method
# initial_size = [100]    # size of initial embedding
# choose = "degrees"     # choose initial nodes by highest degree
# dim = 128     # embedding dimension
# H = None    # put a sub graph if you have an initial sub graph to embed, else None
# SE = StaticEmbeddings(name, G, initial_size, initial_method=initial_method, method=method, H=H, dim=dim, choose=choose, file_tags=file_tags)
# SE.save_embedding(embeddings_path)    # save the embedding
# list_dict_embedding = SE.list_dicts_embedding    # the embedding saved as a dict, this is list of dicts, each dict for different initial size

