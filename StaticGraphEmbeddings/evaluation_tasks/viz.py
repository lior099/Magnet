from visualization import *
from eval_utils import *

datasets_path = os.path.join("..", "datasets")
labels_path = os.path.join("..", "labels")

DATASET1 = {"name": "DBLP", "label_file": "dblp_tags.txt"}
# DATASET2 = {"name": "Cora", "label_file": "cora_tags.txt"}
# DATASET3 = {"name": "Pubmed", "label_file": "pubmed2_tags.txt"}
# DATASET4 = {"name": "Yelp", "label_file": "yelp_labels.txt"}
# DATASET5 = {"name": "Reddit", "label_file": "reddit_tags.txt"}

datasets = [DATASET1]

# choose initial size
initial = 2801

list_keys = []

# choose state-of-the-art methods and our method
methods_ = ["D-VERSE"]
initial_methods_ = ["node2vec"]
mapping = {"VERSE": "OGRE", "D-VERSE": "DOGRE", "We-VERSE": "WOGRE", "LGF": "LGF"}
for j in initial_methods_:
    for m in methods_:
        list_keys.append(j + " + " + m)
if "LGF" in methods_:
    list_keys.remove("HOPE + LGF")
    list_keys.remove("node2vec + LGF")
keys_ours = list_keys
keys_state_of_the_art = initial_methods_
keys = keys_ours  # + keys_state_of_the_art

for i in range(len(datasets)):
    if datasets[i]["name"] == "Pubmed":
        G = nx.read_edgelist(os.path.join(datasets_path, datasets[i]["name"] + ".txt"), create_using=nx.DiGraph(),
                             delimiter=',')
    elif datasets[i]["name"] == "Yelp":
        G = load_data("yelp.npz")
    else:
        G = nx.read_edgelist(os.path.join(datasets_path, datasets[i]["name"] + ".txt"), create_using=nx.DiGraph())
        
    list_in_embd = list(G.nodes())
    
    for key in keys:
        print(key)
        initial_method = key.split(" + ")[0]
        our_method = key.split(" + ")[1]
        X = np.loadtxt(os.path.join("..", "embeddings_degrees",
                                    '{} + {} + {} + {}.txt'.format(datasets[i]["name"], initial_method, our_method,
                                                                   initial)))

        if datasets[i]["name"] != "Yelp":
            labels = read_labels_only_embd_nodes(os.path.join(labels_path, datasets[i]["label_file"]), list_in_embd)
        else:
            labels = []
        our_method = mapping[our_method]
        visualization(datasets[i]["name"], initial_method, our_method, X, list_in_embd, i, labels, initial)
