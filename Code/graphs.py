import random

import networkx as nx
import pickle

class Params:
    def __init__(self, data_name, name, id, path, files, gt, num_of_groups, from_to_ids):
        self.data_name = data_name
        self.name = name
        self.id = id
        self.path = path
        self.num_of_groups = num_of_groups
        self.from_to_ids = from_to_ids
        self.files = files
        self.gt = gt
        if not (name and data_name and id and files):
            raise Exception("Not enough params")
        if self.gt:
            self.gt = self.gt[0]





