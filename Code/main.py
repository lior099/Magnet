
# This version of MAGNET contains all the features from the thesis. BUT this is not a very user friendly version.
# To make it organized and use friendly, more work need to be done.
from datetime import datetime
import os
import random
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from Code.run_tasks import run_task

"""
#### README ####
This file is the main file, with a toy example for a run (toy is a bipartite graph, and toy_multi is a multipartite graph)

here is the list of tasks:

task_num="1" : BipartiteProbabilisticMatchingTask
task_num="2" : MultipartiteCommunityDetectionTask
task_num="3" : PathwayProbabilitiesCalculationTask

task_num="4" : ProbabilitiesUsingEmbeddingsTask (NOT USED in the magnet paper)
task_num="5" : BipartiteNaiveTask (just task 1 with naive algorithm)
task_num="6" : MultipartiteGreedyTask (just task 2 with greedy algorithm)
"""
def main():

    # This is the main file where we run stuff.
    # Here we run an example on the "toy" dataset with 2 shapes and "toy_multi" dataset with 3 shapes.

    # Some initializations
    random.seed(0)
    np.random.seed(0)
    print(f'Time is: {datetime.now()}')
    if 'Code' not in os.listdir(os.getcwd()):
        os.chdir("..")
        if 'Code' not in os.listdir(os.getcwd()):
            raise Exception("Bad pathing, use the command os.chdir() to make sure you work on Magnet directory")
    start = time.time()

    # Here we run on "toy" dataset with 2 shapes (groups).
    data_name = "toy"
    run_task(task_num="1", data_name=data_name, results_root="Results", task_params={'num_of_groups': 2})
    run_task(task_num="2", data_name=data_name, results_root="Results", task_params={'num_of_groups': 2, 'beta': [0.1, 0.1]})
    run_task(task_num="3", data_name=data_name, results_root="Results", task_params={'num_of_groups': 2})
    # Task 4 is not relevant anymore
    # run_task(task_num='4', data_name=data_name, results_root="Results", task_params={'num_of_groups': 2, 'embedding': 'node2vec'})
    run_task(task_num="5", data_name=data_name, results_root="Results", task_params={'num_of_groups': 2})
    run_task(task_num="6", data_name=data_name, results_root="Results", task_params={'num_of_groups': 2})

    # Here we run on "toy_multi" dataset with 3 shapes (groups).
    data_name = "toy_multi"
    run_task(task_num="1", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
    run_task(task_num="2", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3, 'beta': [0.1, 0.1, 0.1]})
    run_task(task_num="3", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
    # Task 4 is not relevant anymore
    # run_task(task_num='4', data_name=data_name, results_root="Results", task_params={'num_of_groups': 2, 'embedding': 'node2vec'})
    run_task(task_num="5", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})
    run_task(task_num="6", data_name=data_name, results_root="Results", task_params={'num_of_groups': 3})



    print('TOTAL TIME:', time.time() - start)
    print(f'Time is: {datetime.now()}')



if __name__ == '__main__':
    main()
