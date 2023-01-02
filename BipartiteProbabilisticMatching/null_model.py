"""
This algorithm is simple:
From the weight matrix, we produce a matrix p.
For each row in w, we build a row in p, in which p_ij = 0 if w_ij is not maximal in row i, and otherwise
p_ij = 1 / (number of maxima).
"""
import networkx as nx
import numpy as np
from scipy import sparse

from BipartiteProbabilisticMatching.matching_solutions import MatchingProblem


def algorithm(mp: MatchingProblem, params=None, is_normalized=None):
    """
    The implementation of the null model algorithm.
    :param mp: The main class containing the graph and its biadjacency matrix, on which we want to apply the algorithm.
    The algorithm uses the biadjacency weight matrix only.
    :param params: We will not use it anyway
    :return: The final probability matrix p.
    """
    w = nx.to_scipy_sparse_matrix(mp.graph)#.toarray()

    p = sparse.csr_matrix((w.shape))
    for i in range(mp.shape[0]):
        if w[i, :].max() == 0:
            continue

        p[i, np.argmax(w[i, :])] = 1
        # s = np.sum(p[i, :])
        # if s:
        #     p[i, :] = np.divide(p[i, :], s)
    # p = nx.to_scipy_sparse_matrix(p)
    p = p[:mp.shape[0], mp.shape[0]:]
    return p
