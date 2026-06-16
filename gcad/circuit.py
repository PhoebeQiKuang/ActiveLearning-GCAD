# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Topo():
    """Directed graph representation of a synthetic gene circuit topology."""

    def __init__(self, edge_list, dose_list, promo_node):
        """Build topology from an edge list, plasmid dose mapping, and promoter node."""
        self.edge_list = edge_list
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edge_list)

        self.promo_node = promo_node  
        self.dose = dose_list
        
        # [NOTE for OED]: The Reporter dose is fixed at 200ng here. 
        # In the future OED, if we want to vary Reporter plasmid, 
        # we might need to make this an argument. For now, matching GCAD defaults is safer.
        self.dose.update({'Rep': 200})
        self.part_list = [k for k in self.dose.keys() if k != 'Rep']
        
        # [NOTE for OED]: These are the nominal degradation rates.
        # Our OED Wrapper will (likely) override these in the ODE solver functions,
        # but we keep them here as placeholders to prevent the "Miner" from breaking.
        self.protein_deg = {'Z': 0.35, 'I': 0.35, 'R': 0.029}

        self.in_dict = dict() # Classify nodes
        self.pool = dict()
        for n in (self.part_list + ['Rep']):
            pre = list(self.graph.predecessors(n))
            # for each part, dict with each predecessor
            # sorted by type of part
            self.in_dict.update({n:
                                     {'P': [i for i in pre if i[0] == 'P'],
                                      'Z': [i for i in pre if i[0] == 'Z'],
                                      'I': [i for i in pre if i[0] == 'I']}})
            # for each part, determine whether dose
            # must be divided between P and Z (if 
            # both regulators in in_dict) or entire
            # pool used for one regulator
            self.pool.update({n: (self.in_dict[n]['P'] != []) + (self.in_dict[n]['Z'] != [])})
        if 0 in list(self.pool.values()):
            raise Exception("Something's wrong. No activator in the circuit.")

        self.num_states = len(self.in_dict.keys())
        self.var_dict = dict(zip((self.in_dict.keys()), np.arange(self.num_states)))
        self.valid = None

    def check_valid(self):
        """Validate that every part has an activator input and a path to Rep."""
        self.valid = 1
        for n in self.part_list:
            if self.graph.in_degree(n) <= len(self.in_dict[n]['I']):
                self.valid = 0
            if len(list(nx.all_simple_paths(self.graph, n, 'Rep'))) == 0:
                self.valid = 0

    def update(self, edge_list):
        """Rebuild graph and all internal state after a mutation."""
        self.edge_list = edge_list
        self.graph = nx.DiGraph(self.edge_list)
        self.part_list = [k for k in self.dose.keys() if k != 'Rep']

        self.in_dict = dict()  # Classify nodes
        self.pool = dict()
        for n in (self.part_list + ['Rep']):
            pre = list(self.graph.predecessors(n))
            self.in_dict.update({n:
                                     {'P': [i for i in pre if i[0] == 'P'],
                                      'Z': [i for i in pre if i[0] == 'Z'],
                                      'I': [i for i in pre if i[0] == 'I']}})
            self.pool.update({n: (self.in_dict[n]['P'] != []) + (self.in_dict[n]['Z'] != [])})
        if 0 in list(self.pool.values()):
            raise Exception("Something's wrong. No activator in the circuit.")

        self.num_states = len(self.in_dict.keys())
        self.var_dict = dict(zip((self.in_dict.keys()), np.arange(self.num_states)))

    def plot_graph(self):
        """Draw the circuit as a directed graph (useful for debugging)."""
        plt.figure()
        plt.tight_layout()
        nx.draw_networkx(self.graph, arrows=True, arrowsize=15, node_size=600, node_shape='s')
        plt.show()