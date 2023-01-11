
"""
#### README ####
This file is for plotting. Notice some imports are missing, if you want to use them read exactly what they do.
"""
#

def plot_graph(data_paths, colors, x_label, y_label, log=False, log_x=False, save_path=None, my_labels=None, title='', axis=None, invert_labels=False):
    title_size = 13
    x_label_size = 10
    if axis:
        plt = axis
    x_list, y_list = [], []
    for path in data_paths:
        with open(path, 'r', newline='') as file:
            data = list(csv.reader(file))
            x_list.append([float(i) for i in data[0][1:]])
            y_list.append([float(i) for i in data[1][1:]])
            # labels.append(data[1][0])
    if my_labels:
        labels = my_labels
    if invert_labels:
        labels[0] = 0.00001
        labels, x_list = x_list[0], [labels for a in x_list]
        labels = [int(a) if a > 1 else a for a in labels]
        labels = [f'{x_label}={a}' for a in labels]
        x_label = 'Beta'
        y_list = np.array(y_list).T.tolist()
        print("e")
    for x, y, color, label in zip(x_list, y_list, colors, labels):
        x, y = zip(*sorted(zip(x, y)))
        plt.plot(x, y, "-o", color=color, label=label)
    if x_label:
        plt.set_xlabel(x_label, fontsize=x_label_size) if axis else plt.xlabel(x_label, fontsize=x_label_size)
    if y_label:
        plt.set_ylabel(y_label) if axis else plt.ylabel(y_label)
    # plt.xticks(np.arange(0.0, 0.6, 0.1))
    # plt.xscale('log')
    if log:
        plt.set_yscale('log') if axis else plt.yscale('log')
        # plt.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        # plt.yaxis.set_major_formatter(mticker.ScalarFormatter())
    if log_x:
        plt.set_xscale('log') if axis else plt.xscale('log')
    plt.tick_params(axis="x", labelsize=8, which='both')
    plt.tick_params(axis="y", labelsize=8, which='both')
    plt.set_title(title, fontsize=title_size) if axis else plt.title(title, fontsize=title_size)
    plt.grid(True, linestyle='--', which="both")
    # matplotlib.rcParams.update({'font.size': 100})
    # matplotlib.rc('xtick', labelsize=50)
    if invert_labels:
        plt.legend(fontsize=7)
    if axis:
        return
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()


def plot_results(results, colors, x_label, y_label, graph_name, labels=None, title='', axis=None, log=False, log_x=False, invert_labels=False):
    paths = []
    task_names = {'1': 'task_1_BipartiteProbabilisticMatching',
                  '2': 'task_2_MultipartiteCommunityDetection',
                  '3': 'task_3_PathwayProbabilitiesCalculation',
                  '4': 'task_4_ProbabilitiesUsingEmbeddings',
                  '5': 'task_5_BipartiteNaive',
                  '6': 'task_6_MultipartiteGreedy'}
    for result in results:
        path = SEP.join(["Results", task_names[result["task"]], result["name"], result["graph"] + '.csv'])
        paths.append(path)
    save_path = SEP.join(["Results", graph_name + ".png"])
    plot_graph(paths, colors, log=log, log_x=log_x, x_label=x_label, y_label=y_label, save_path=save_path, my_labels=labels, title=title, axis=axis, invert_labels=invert_labels)
    # plot_graph(paths, colors, x_label="Fraction", y_label="Running Time[s]", save_path=save_path)
    # plot_graph([memory_destination], ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")




def plot_all_results1():
    print(f"Plotting Results! os.getcwd(): {os.getcwd()}")
    # [false_mass, nodes, noisy_edges, removed_nodes, restaurant, test, Abt - Buy]
    # [avg_acc, f1_score, memory, runtime, top5_acc, winner_acc, best_f1]
    # plot_graph(runtime_destination, ["blue"], x_label="Fraction", y_label="Running Time[s]",
    #            save_path=runtime_destination[:-4] + ".png")
    # plot_graph(memory_destination, ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")
    # runtime_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_runtime.csv'
    # memory_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_memory.csv'

    results = [{'task': '1', 'name': 'false_mass', 'graph': 'winner_acc'},
               {'task': '1', 'name': 'noisy_edges', 'graph': 'winner_acc'},
               {'task': '1', 'name': 'removed_nodes', 'graph': 'winner_acc'}]
    plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_1_winner_accuracy',
                 ['False Mass', 'Noisy Edges', 'Removed Nodes'], '')




    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'ogre_avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
    #              ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'ogre_winner_accuracy'}]
    # plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
    #              ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass winner accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'node2vec_top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'ogre_top5_accuracy'}]
    # plot_results(results, ['blue', 'red', 'black'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
    #              ['task 3', 'task 4 node2vec', 'task 4 OGRE'], 'Tasks 3 vs 4 false_mass top-5 accuracy')

    # ---
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_avg_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes avg accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_avg_norm_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_winner_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes winner accuracy')
    #
    # results = [{'task': '3', 'name': 'nodes', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'nodes', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Nodes', 'Accuracy', 'tasks_3_vs_4_nodes_top5_accuracy',
    #              ['task 3', 'task 4 node2vec'], 'Tasks 3 vs 4 nodes top-5 accuracy')
    # ---
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass winner accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass top-5 accuracy')
    # ---
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'avg_norm_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_avg_norm_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass avg norm accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'winner_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'winner_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_winner_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass winner accuracy')
    #
    # results = [{'task': '3', 'name': 'false_mass', 'graph': 'top5_accuracy'},
    #            {'task': '4', 'name': 'false_mass', 'graph': 'top5_accuracy'}]
    # plot_results(results, ['blue', 'red'], 'Fraction', 'Accuracy', 'tasks_3_vs_4_false_mass_top5_accuracy',
    #              ['task 3', 'task 4'], 'Tasks 3 vs 4 false mass top-5 accuracy')

    # results = [{'task': '1', 'name': 'false_mass', 'graph': 'runtime'}]
    # colors = ['blue']
    # plot_results(results, colors, 'Fraction', 'Running Time[s]', 'task_1_false_mass_nodes_runtime')

    # results = [{'task': '1', 'name': 'false_mass', 'graph': 'memory'}]
    # colors = ['blue']
    # plot_results(results, colors, 'Fraction', 'Memory Usage[Mb]', 'task_1_false_mass_memory')

def plot_all_results2(save_file='results.png'):
    print(f"Plotting Results! os.getcwd(): {os.getcwd()}")
    datasets = [
                {'name': 'nodes', 'x_label': 'Vertices', 'title': 'Number of Vertices'},
                {'name': 'noisy_edges', 'x_label': 'Fraction', 'title': 'Noisy Edges'},
                {'name': 'false_mass', 'x_label': 'Fraction', 'title': 'False Mass'},
                {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Removed Vertices'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Number of Vertices'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Noisy Edges'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'False Mass'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Removed Real Vertices'}
        ]
    tasks = [
             # {'task': '5', 'acc': 'winner_acc', 'color': 'blue', 'label': 'Greedy'},
             {'task': '6', 'acc': 'all_avg_acc', 'color': 'blue', 'label': 'Greedy'},
             # {'task': '1', 'acc': 'winner_acc', 'color': 'red', 'label': 'BPM'},
             {'task': '2', 'acc': 'all_avg_acc', 'color': 'black', 'label': 'MCD'},
             # {'task': '3', 'acc': 'winner_acc', 'color': 'green', 'label': 'PPC'},
             ]
    figure, axis = plt.subplots(3, 4)
    colors = [task['color'] for task in tasks]
    labels = [task['label'] for task in tasks]
    for i, dataset in enumerate(datasets):
        y_label = 'Accuracy' if i == 0 else ''
        results = [{'task': task['task'], 'name': dataset['name'], 'graph': task['acc']} for task in tasks]
        # results = [{'task': '5', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '1', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '2', 'name': dataset['name'], 'graph': 'all_avg_acc'},
        #            {'task': '3', 'name': dataset['name'], 'graph': 'winner_acc'}]
        plot_results(results, colors, '', y_label, f'{dataset["name"]}_accuracy',
                     labels, dataset['title'], axis[0, i])

        y_label = 'Running Time [s]' if i == 0 else ''
        results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'runtime'} for task in tasks]
        plot_results(results, colors, '', y_label, f'{dataset["name"]}_runtime',
                     labels, '', axis[1, i], log=True)

        y_label = 'Memory [MiB]' if i == 0 else ''
        results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'memory'} for task in tasks]
        plot_results(results, colors, dataset['x_label'], y_label, f'{dataset["name"]}_memory',
                     labels, '', axis[2, i], log=True)
    axis[1, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axis[0, i].legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0.4)
    # plt.tight_layout()
    plt.savefig(SEP.join(["Results", save_file]))
    plt.show()

def plot_all_results_task_2(save_file='results_task_2_v1.png'):
    print(f"Plotting Results! os.getcwd(): {os.getcwd()}")
    datasets = [
                {'name': 'nodes', 'x_label': 'Vertices', 'title': 'Number of Vertices'},
                {'name': 'noisy_edges', 'x_label': 'Fraction', 'title': 'Noisy Edges'},
                {'name': 'false_mass', 'x_label': 'Fraction', 'title': 'False Mass'},
                {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Removed Vertices'},

                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Number of Vertices'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Noisy Edges'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'False Mass'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Removed Real Vertices'}
        ]
    # ids = list(np.arange(1.8, 0.3, -0.2))
    # ids = [0.1] + ids[::-1]
    ids = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0.1, 0.7, len(ids))]
    colors = ['blue', 'red', 'black', 'green', 'orange', 'purple']
    tasks = [{'task': '2', 'acc': f'{i}_beta_all_avg_acc', 'color': color, 'label': f'beta={i}'} for i, color in zip(ids, colors)]
    figure, axis = plt.subplots(2,2)
    labels = [task['label'] for task in tasks]
    for i, dataset in enumerate(datasets):
        x_label, y_label = None, None
        if i in [0]:
            x_label = 'Vertices'
        else:
            x_label = 'Fraction'
        if i in [0, 2]:
            y_label = 'Accuracy'
        results = [{'task': task['task'], 'name': dataset['name'], 'graph': task['acc']} for task in tasks]
        # results = [{'task': '5', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '1', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '2', 'name': dataset['name'], 'graph': 'all_avg_acc'},
        #            {'task': '3', 'name': dataset['name'], 'graph': 'winner_acc'}]
        plot_results(results, colors, x_label, y_label, f'{dataset["name"]}_accuracy',
                     labels, dataset['title'], axis.flatten()[i])

        # y_label = 'Running Time [s]' if i == 0 else ''
        # results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'runtime'} for task in tasks]
        # plot_results(results, colors, '', y_label, f'{dataset["name"]}_runtime',
        #              labels, '', axis[1, i], log=True)
        #
        # y_label = 'Memory [MiB]' if i == 0 else ''
        # results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'memory'} for task in tasks]
        # plot_results(results, colors, dataset['x_label'], y_label, f'{dataset["name"]}_memory',
        #              labels, '', axis[2, i], log=True)
    axis[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axis[0, i].legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0.4)
    # plt.tight_layout()
    plt.savefig(SEP.join(["Results", save_file]))
    plt.show()

def plot_all_results_task_2_v2(save_file='results_task_2_v2.png'):
    print(f"Plotting Results! os.getcwd(): {os.getcwd()}")
    datasets = [
                {'name': 'nodes', 'id': 'Vertices', 'title': 'Number of Vertices'},
                {'name': 'noisy_edges', 'id': 'Fraction', 'title': 'Noisy Edges'},
                {'name': 'false_mass', 'id': 'Fraction', 'title': 'False Mass'},
                {'name': 'removed_nodes', 'id': 'Fraction', 'title': 'Removed Vertices'},

                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Number of Vertices'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Noisy Edges'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'False Mass'},
                # {'name': 'removed_nodes', 'x_label': 'Fraction', 'title': 'Removed Real Vertices'}
        ]
    # ids = list(np.arange(1.8, 0.3, -0.2))
    # ids = [0.1] + ids[::-1]
    ids = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    # cmap = plt.get_cmap('plasma')
    # colors = [cmap(i) for i in np.linspace(0.1, 0.7, len(ids))]
    colors = ['blue', 'red', 'black', 'green', 'orange']
    tasks = [{'task': '2', 'acc': f'{i}_beta_all_avg_acc', 'label': i} for i in ids]
    figure, axis = plt.subplots(2,2)
    labels = [task['label'] for task in tasks]
    for i, dataset in enumerate(datasets):
        x_label, y_label = None, None
        if i in [0]:
            x_label = 'Vertices'
        else:
            x_label = 'Fraction'
        if i in [0, 2]:
            y_label = 'Accuracy'
        results = [{'task': task['task'], 'name': dataset['name'], 'graph': task['acc']} for task in tasks]
        # results = [{'task': '5', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '1', 'name': dataset['name'], 'graph': 'winner_acc'},
        #            {'task': '2', 'name': dataset['name'], 'graph': 'all_avg_acc'},
        #            {'task': '3', 'name': dataset['name'], 'graph': 'winner_acc'}]
        plot_results(results, colors, x_label, y_label, f'{dataset["name"]}_accuracy',
                     labels, dataset['title'], axis.flatten()[i], log_x=True, invert_labels=True)

        # y_label = 'Running Time [s]' if i == 0 else ''
        # results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'runtime'} for task in tasks]
        # plot_results(results, colors, '', y_label, f'{dataset["name"]}_runtime',
        #              labels, '', axis[1, i], log=True)
        #
        # y_label = 'Memory [MiB]' if i == 0 else ''
        # results = [{'task': task['task'], 'name': dataset['name'], 'graph': 'memory'} for task in tasks]
        # plot_results(results, colors, dataset['x_label'], y_label, f'{dataset["name"]}_memory',
        #              labels, '', axis[2, i], log=True)
    # axis[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axis[0, i].legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0.4)
    # plt.tight_layout()
    plt.savefig(SEP.join(["Results", save_file]))
    plt.show()
