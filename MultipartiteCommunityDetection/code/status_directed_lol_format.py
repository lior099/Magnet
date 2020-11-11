class Status_Lol_Format(object):
    def __init__(self):
        self.node2com = {}       # Dictionary {node: community index for every node in the graph}
        self.com_nodes = {}      # Dictionary {community: [count of nodes of each type in this community]}
        self.total_weight = 0    # Sum of edge weights over all edges (including loops)
        self.in_degrees = {}     # Dictionary {community: sum of in-degrees of all nodes in this community}
        self.out_degrees = {}    # Dictionary {community: sum of out-degrees of all nodes in this community}
        self.g_in_degrees = {}   # Dictionary {node: in-degree of a node for every node in the graph}
        self.g_out_degrees = {}  # Dictionary {node: out-degree of a node for every node in the graph}
        self.internals = {}      # Dictionary {community: sum of all degrees of edges within the community}
        self.loops = {}          # Dictionary {node: degree of self-edge if exists, else 0}

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status_Lol_Format()
        new_status.node2com = self.node2com.copy()
        new_status.com_nodes = self.com_nodes.copy()
        new_status.internals = self.internals.copy()
        new_status.in_degrees = self.in_degrees.copy()
        new_status.out_degrees = self.out_degrees.copy()
        new_status.g_in_degrees = self.g_in_degrees.copy()
        new_status.g_out_degrees = self.g_out_degrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, nodetype, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = {}
        self.in_degrees = {}
        self.out_degrees = {}
        self.g_in_degrees = {}
        self.g_out_degrees = {}
        self.internals = {}
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                in_deg = round(float(graph.in_degree(node, weight=weight)), 5)
                out_deg = round(float(graph.out_degree(node, weight=weight)), 5)
                com_type = graph.return_node_type(node, groups=3)
                if any([in_deg < 0, out_deg < 0]):
                    raise ValueError(f"Bad node degree for node ({node})")
                self.com_nodes[count] = com_type
                self.in_degrees[count] = in_deg
                self.out_degrees[count] = out_deg
                self.g_in_degrees[node] = in_deg
                self.g_out_degrees[node] = out_deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                in_deg = float(graph.in_degree(node))
                out_deg = float(graph.in_degree(node))
                com_type = graph.return_node_type(node, groups=3)
                if com not in self.com_nodes:
                    self.com_nodes[com] = com_type
                else:
                    self.com_nodes[com] = [exist + add for exist, add in zip(self.com_nodes[com], com_type)]
                self.in_degrees[com] = self.in_degrees.get(com, 0) + in_deg
                self.out_degrees[com] = self.out_degrees.get(com, 0) + out_deg
                self.g_in_degrees[node] = in_deg
                self.g_out_degrees[node] = out_deg
                inc = 0.
                for neighbor, datas in graph.graph_adjacency[node]:
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        raise ValueError(f"Bad edge weight ({edge_weight})")
                    if part[neighbor] == com:
                        inc += float(edge_weight)
                self.internals[com] = self.internals.get(com, 0) + inc
