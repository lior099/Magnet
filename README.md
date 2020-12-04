# Bipartite Probabilistic Matching and Multipartite Community Detection and Pathway Probabilities Calculation 
Shoval Frydman, Kfir Salomon, Itay Levinas, Haim Isakov, Lior Shifman.

## Problem Description
1. Given a weighted bipartite graph, we calculate a probability matrix ***P***, such that ***P[i, j]*** represents the 
probability for the node ***i*** to match the node ***j***.  
Note that this matrix is normalized over the rows only. 
2. Given a directed, weighted multipartite graph (with weights that could be the weights obtained from the first 
solution), we divide the graph into communities, aiming that few nodes from each side will be in the same community.
3. Given a directed, weighted multipartite graph (with weights that could be the weights obtained from the first 
solution), we calculate the probability to get from a starting point to all vertices that are reachable from starting point.

## Suggested Algorithms
For the first problem, we suggest three algorithms (null model, flow and updating by degree) to solve this problem.  
The second problem is solved by a variation of the Louvain algorithm.  

     

The algorithms are implemented in the code directory of each algorithm (the flow algorithm has two implementations - analytic and 
numeric). The Louvain code implementation is based on the 
[python-louvain](https://python-louvain.readthedocs.io/en/latest/) package.

## How to Run
-T task-number -S data-source-directory -D data-saving-directory  
For example:
```
-T 3 -S yoram_network_4 -D yoram_network_4
```
For each task we want to run, we first store the data in AlgoName/data/data-source-directory and the results will be in AlgoName/results/data-saving-directory.  
Note: passway probabilitis calculation receives normalized multi-partite graph, so it is recommended to run the Bipartite Probabilistic Matching first.  
 
