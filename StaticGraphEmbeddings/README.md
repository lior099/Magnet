# StaticGraphEmbeddings
This repository presents several online static graph embedding methods suggested by us. Full description and survey of the static graph embedding problem, current work, our
suggested methods, tasks for evaluation and results can be found in the file: "Static Graph Embedding.pdf".

## Our code
1. The directory "our_embeddings_methods" contains implementations of our 4 suggested static embedding methods: OGRE, DOGRE, WOGRE and LGF. All implementations are well
documented. The main file in this directory is "static_embeddings.py" which calculates a graph embedding with one of our suggested methods. Running instructions can be found there.
2. The directory "state_of_the_art" contains implementations of 3 state-of-the-art methods we have used both as part of our methods and for comparison- node2vec, HOPE 
and Graph Factorization.
3. The directory "evaluation_tasks" contains implementations of three evaluation tasks: Link Prediction, Node Classification and Visualization. In addition, it contains the file 
"calculate_static_embeddings.py" which is an example file that applies our suggested methods on the given dataset and evaluates their preformance. Each file in this directory
contains a full description of what it does and how to run it.

## Datasets
The directory "datasets" contains three example datasets. They can be undirected/directed, weighted/unweighted. We assume they come as an edgelist, i.e. a text file with two 
columns if the graph is unweighted (source target - 1 2) or three columns if the graph is weighted (source target weight - 1 2 0.9). The graph does not have to be labeled in order
to calculate its embedding, but notice that for node classification task labels are must.

For more information and questions, one can contact us in this mail address: 123shovalf@gmail.com
