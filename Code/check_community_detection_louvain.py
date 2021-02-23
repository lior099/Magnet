import csv
import os


def check_communities(file):
    community_dict = {}
    with open(file, "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)  # skip the headers
        next(datareader, None)
        for triple in datareader:
            com = int(triple[2])
            type = int(triple[1])
            if com not in community_dict:
                community_dict[com] = {0: 0, 1: 0, 2: 0, 3: 0}
            community_dict[com][type] = community_dict[com][type] + 1
            next(datareader, None)
    return community_dict


if __name__ == '__main__':
    rootDir = os.path.join("..", "MultipartiteCommunityDetection", "results")
    filenames = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in
                              os.walk(rootDir) for file in filenames]
    for file in filenames:
        print(file)
        print()
        print(check_communities(file))
