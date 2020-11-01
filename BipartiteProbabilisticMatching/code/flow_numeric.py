"""
An numeric algorithm for solving the matching problem, using the following algorithm:
Input: A weighted bipartite graph.

1. Let W be the ("full") weighted adjacency matrices of the bipartite graph.
   W will be normalized and represent a directed graph, such that for every vertex with output degree >= 1
   the sum of all weights of edges from it will be 1.
2. Let p be a vector of the amount of fluids held in each vertex. It is initialized as a vector of ones, of size equal
   to the number of vertices.
3. The vector p is iteratively updated until convergence (i.e., the 1-norm of the step, p_new - p_old, is smaller than a
   parameter epsilon). The step of updating is done by flowing on the edges according to the following formula:
                        p = rho_0 * p + rho_1 (p * w) + (1 - rho_0 - rho_1) ones_vector
   where rho_0, rho_1 are parameters of the model, ones_vector is a vector of ones with the same shape as p.
4. Using p and W, we create a matrices of contributions, where the (i, j)-th element represents the amount of fluid that
   j receives from i per time unit, according to the converged vector p.
5. The contributions matrices is normalized like in part 1, resulting the probabilities matrices.
   Here, we take the biadjacency matrices which is the upper right block for calculating the desired measurements.
   However, the entire (non-symmetric) matrices is calculated.
Output: The required probabilities matrices.


NOTE: The function "algorithm", which is the main function of this algorithm, receives a dictionary named 'params'
of the following parameters (e.g. {'rho_0': 0.6, ...}):
    rho_0 - Controls the amount of fluid we choose to keep in each vertex (rather than flow to its neighbors) per time
            unit.
    rho_1 - Controls the amount of fluid we choose to pass on the edges. (1 - rho_0 - rho_1) represents the amount of
            fluid per time unit added to each vertex.
    epsilon - The tolerance constant. Controls how close we want to converge to the analytic solution.
"""
import networkx as nx
import numpy as np
from BipartiteProbabilisticMatching.code.matching_solutions import MatchingProblem
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, linalg, lil_matrix


def algorithm(mp: MatchingProblem, params, is_normalized=True):
    """
    The main function of the algorithm.

    :param mp: The main class containing the graph on which we want to apply the algorithm.
    :param params: The dictionary of the parameters of this algorithm.
    :param is_normalized: Whether to normalize the contribution matrix by rows.
    :return: The final probability matrices.
    """
    w = nx.to_scipy_sparse_matrix(mp.graph)
    w = normalization(w)
    p_node_balance = node_balance(w, params['rho_0'], params['rho_1'], params['epsilon'])
    contribution_matrix = calculate_contribution_matrix(p_node_balance, w, mp.shape)
    if is_normalized:
        c = normalization(contribution_matrix)
    else:
        c = contribution_matrix
    return c


def normalization(m):
    """
    Normalize the matrices by rows. The normalization is set such that for all vertices with positive degree (meaning the
    sum of the row is not zero), the sum of m over the corresponding row will become 1.

    :param m: The adjacency matrices.
    :return: The normalized adjacency matrices by rows.
    """
    w_normalized = normalize(m, norm='l1', axis=1)
    return w_normalized


def node_balance(w, r0, r1, epsilon):
    """
    Create the vector p (initialized as a vector of ones) and run the flow until convergence.

    :param w: The normalized full adjacency matrices.
    :param r0: The parameter rho_0.
    :param r1: The parameter rho_1.
    :param epsilon: The tolerance parameter. We will stop the iterations if ||p_new - p_old||_1 < epsilon.
    :return: The final (steady state) vector p.
    """
    p = csr_matrix(np.ones(w.shape[0]))
    condition = epsilon + 1
    ones = csr_matrix(np.ones(p.shape[1]))
    while condition > epsilon:
        p_new = r0 * p + r1 * p.dot(w) + (1 - r0 - r1) * ones
        condition = linalg.norm(p_new-p, 1)
        p = p_new
    return p


def calculate_contribution_matrix(p, w, shape):
    """
    Using p vector and w, we calculate the contribution matrices: c_ij = p_i * w_ij / p_j
    where c_ij is the contribution of vertex i to the vertex j, as explained above.

    :param p: The vector p of amounts of fluids in steady state.
    :param w: The normalized full adjacency matrices.
    :param shape: The shape of the biadjacency matrix.
    :return: The contribution matrices
    """
    p = lil_matrix(p)
    w = lil_matrix(w)
    m = lil_matrix((shape[0], shape[1]))
    a = p.copy()
    p_inv = lil_matrix((p.shape[0], shape[1]))
    p = p[0, shape[1]:]
    w = w[:shape[0], shape[0]:]
    for j in range(shape[1]):
        p_inv[0, j] = 1/a[0, shape[0] + j]
    for i in range(shape[0]):
        p_w = p[0, i] * w[i, :]
        d = p_w.multiply(p_inv[0, :])
        m[i, :] = d
    return m
