"""
Node classification Task For Evaluation
"""


try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from StaticGraphEmbeddings.state_of_the_art.state_of_the_art_embedding import *


"""
Code for the node classification task as explained in GEM article. This part of the code belongs to GEM.
For more information, you can go to our github page.
"""


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction


def evaluateNodeClassification(X, Y, test_ratio):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro and accuracy.
    """
    number_of_labels = Y.shape[1]
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
        X,
        Y,
        test_size=test_ratio
    )
    index = []
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr(solver='lbfgs', max_iter=350000))
    classif2.fit(X_train, Y_train)
    prediction = classif2.predict(X_test, top_k_list)
    accuracy = accuracy_score(Y_test, prediction)
    micro = f1_score(Y_test, prediction, average='micro', zero_division=0)
    macro = f1_score(Y_test, prediction, average='macro', zero_division=0)
    auc = roc_auc_score(Y_test, prediction)
    return micro, macro, accuracy, auc


def expNC(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average
    :return: Scores for all splits and all splits- F1-micro, F1-macro and accuracy.
    """
    micro = [None] * rounds
    macro = [None] * rounds
    acc = [None] * rounds
    auc = [None] * rounds

    for round_id in range(rounds):
        micro_round = [None] * len(test_ratio_arr)
        macro_round = [None] * len(test_ratio_arr)
        acc_round = [None] * len(test_ratio_arr)
        auc_round = [None] * len(test_ratio_arr)

        for i, test_ratio in enumerate(test_ratio_arr):
            micro_round[i], macro_round[i], acc_round[i], auc_round[i] = evaluateNodeClassification(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round
        auc[round_id] = auc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)
    auc = np.asarray(auc)

    return micro, macro, acc, auc


def calculate_avg_score(score, rounds):
    """
    Given the lists of scores for every round of every split, calculate the average score of every split.
    :param score: F1-micro / F1-macro / Accuracy
    :param rounds: How many times the experiment has been applied for every split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores_for_all(micro, macro, acc, auc, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro = calculate_avg_score(micro, rounds)
    all_avg_macro = calculate_avg_score(macro, rounds)
    all_avg_acc = calculate_avg_score(acc, rounds)
    all_avg_auc = calculate_avg_score(auc, rounds)
    return all_avg_micro, all_avg_macro, all_avg_acc, all_avg_auc


def read_labels(name, file_tags, dict_proj, mapping=None):
    """
    Read the labels file and return the labels as a matrix. Matrix is from size number of samples by number
    of labels, where C[i,j]==1 if node i has label j, else 0.
    :param file_tags: a file with labels for every node
    :return: matrix as explained above
    """
    if name == "Yelp":
        Y, dict_proj = read_yelp_labels(file_tags, mapping, dict_proj)
    else:
        if name == "Reddit":
            f = open(file_tags, 'r')
            labels = {}
            for line in f:
                name = line.split(" ")[0]
                label = int(line.split(" ")[1].split("\n")[0])
                labels.update({name: label})
            f.close()
        else:
            c = np.loadtxt(file_tags).astype(int)
            if name == "ER-AvgDeg10-1M-L2":
                labels = {str(x): int(y - 1) for (x, y) in c}
            elif name == "Pubmed":
                labels = {str(x): int(y - 1) for (x, y) in c}
            else:
                labels = {str(x): int(y) for (x, y) in c}
        keys = list(dict_proj.keys())
        # keys = list(labels.keys())
        values = list(labels.values())
        values = list(dict.fromkeys(values))
        values.sort()
        number_of_labels = values[-1] + 1
        Y = np.zeros((len(dict_proj), number_of_labels))
        for i in range(len(keys)):
            key = keys[i]
            # key = int(keys[i])
            # tag = labels[str(key)]
            tag = labels[str(key)]
            for j in range(number_of_labels):
                if j == tag:
                    Y[i, j] = 1
        for k in range(number_of_labels):
            if np.all((Y[:, k] == 0), axis=0):
                Y = np.delete(Y, k, 1)
    return Y, dict_proj


def read_yelp_labels(file_tags, mapping, dict_proj):
    """
    Read labels of yelp dataset
    """
    X = np.loadtxt(file_tags)
    Y = np.int_(X)
    number_of_labels = Y.shape[1]
    for k in range(number_of_labels):
        if np.all((Y[:, k] == 0), axis=0):
            Y = np.delete(Y, k, 1)
    not_here = len(dict_proj) - Y.shape[0]
    for n in range(not_here):
        del dict_proj[mapping[n]]
    return Y, dict_proj


def our_embedding_method(dict_proj, dim):
    """
    Run cheap node2vec and make it a features matrix- matrix of size number of sample by number of embedding
    dimension, where the i_th row of X is its projection from cheap node2vec.
    :param dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a matrix as explained above
    """
    X = np.zeros((len(dict_proj), dim))
    keys = list(dict_proj.keys())
    for i in range(len(keys)):
        X[i, :] = dict_proj[keys[i]]
    return X


def nc_mission(name, key, z, ratio_arr, label_file, dim, rounds, mapping=None):
    """
    Node Classification Task where one wants the scores as a function of size of the initial embedding. Notice test
    ratio must be fixed. The variable that changes here is the size of the initial embedding. For more  explanation,
    see our pdf file attached in out git.
    :param The applied embedding method
    :param z: Embedding dictionary of the given graph (with all types of our methods, no state-of-the-art))
    :param ratio_arr: Test ratio
    :param label_file: File with labels of the graph. For true format see "results_all_datasets.py" file.
    :param dim: Dimension of the embedding space
    :param rounds: How many time to repeat the task for evaluation
    :return: Scores of node classification task for each dataset- Micro-F1, Macro-F1, Accuracy and AUC. They return as
            lists for each size of initial embedding for each method
    """
    dict_initial = {}
    for r in ratio_arr:
        all_micro = []
        all_macro = []
        all_acc = []
        all_auc = []
        if " + " in key:
            list_dict_projections = z[key].list_dicts_embedding
        else:
            list_dict_projections = [z[key][1]]
        for j in range(len(list_dict_projections)):
            Y, dict_proj = read_labels(name, label_file, list_dict_projections[j], mapping)
            X = our_embedding_method(dict_proj, dim)
            micro, macro, acc, auc = expNC(X, Y, [r], rounds)
            avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores_for_all(micro, macro, acc, auc, rounds)
            print(avg_micro)
            print(avg_macro)
            print(avg_acc)
            print(avg_auc)
            all_micro.append(avg_micro[0])
            all_macro.append(avg_macro[0])
            all_acc.append(avg_acc[0])
            all_auc.append(avg_auc[0])
        dict_initial.update({r: [all_micro, all_macro, all_acc, all_auc]})
    return dict_initial


def final_node_classification(name, dict_all_embeddings, params_nc, dict_dataset, mapping=None):
    """
    Node Classification Task
    :param dict_all_embeddings: Dictionary with all dict embeddings for all applied embedding method
    :param params_nc: Parameters for node classification task
    :return: Dict where keys are applied methods and keys are dicts of scores for each test ratio.
    """
    dict_nc_mission = {}

    ratio_arr = params_nc["test_ratio"]
    rounds = params_nc["rounds"]

    keys = list(dict_all_embeddings.keys())

    for key in keys:
        label_file = dict_dataset["label_file"]
        d = dict_dataset["dim"]
        dict_initial = nc_mission(name, key, dict_all_embeddings, ratio_arr, label_file, d, rounds, mapping=mapping)
        dict_nc_mission.update({key: dict_initial})
    return dict_nc_mission
