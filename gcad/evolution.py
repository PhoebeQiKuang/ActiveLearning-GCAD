# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import networkx as nx
import numpy as np
from gcad.circuit import Topo
from copy import deepcopy
from gcad.loader import tf_list, inhibitor_list, parts
from itertools import combinations, permutations


def get_out_path(n, part_list, circuit_tf_list):
    """Build a random downstream edge path from part n to the reporter."""
    out_list = [k for k in part_list if k != n]
    out_path = [n]

    if len(out_list) > 0:
        num_connect = np.random.randint(len(out_list)+1)
        out_path.extend(np.random.choice(out_list, num_connect, replace=False))

    need_z_reg = []
    for i, j in zip(out_path[:-1], out_path[1:]):
        if i[0] == "I":
            need_z_reg.append(j)

    z_added_edges = []
    for i in need_z_reg:
        z_reg = np.random.choice(circuit_tf_list)
        z_added_edges.append((z_reg, i))

    out_path.append('Rep')
    edges = [(i, j) for i, j in zip(out_path[:-1], out_path[1:])]
    edges.extend(z_added_edges)

    return edges


def get_in_path(n, promo_node, circuit_tf_list):
    """Build a random upstream edge path from the promoter to part n."""
    if promo_node is not None:
        in_node = np.random.choice(circuit_tf_list + [promo_node])
        edges = [(in_node, n)]

        if in_node != promo_node:
            in_path = [promo_node]
            num_connect = np.random.randint(len(circuit_tf_list)+1)
            in_path.extend(np.random.choice(circuit_tf_list, num_connect, replace=False))
            in_path.append(n)
            edges.extend([(i, j) for i, j in zip(in_path[:-1], in_path[1:])])

    else:
        in_node = np.random.choice(circuit_tf_list)
        edges = [(in_node, n)]
        if in_node == n:
            in_node = np.random.choice([k for k in circuit_tf_list if k != n])
            edges.append((in_node, n))

    return edges


def get_edges(promo_node, part_list):
    """Assemble a valid random edge list for a given part list and promoter."""
    circuit_tf_list = []
    same_list = []
    for k in part_list:
        if k[0] == 'Z':
            circuit_tf_list.append(k)
            if ('I' + k[1:]) in part_list:
                same_list.append([k, ('I' + k[1:])])

    edge_list = [(promo_node, np.random.choice(circuit_tf_list)),
                 (np.random.choice(circuit_tf_list),'Rep')
    ]

    for n in part_list:
        if not any(n in sublist for sublist in same_list):
            in_edges = get_in_path(n, promo_node, circuit_tf_list)
            for edge in in_edges:
                if edge not in edge_list:
                    edge_list.append(edge)

            out_edges = get_out_path(n, part_list, circuit_tf_list)
            for edge in out_edges:
                if edge not in edge_list:
                    edge_list.append(edge)

    for z, i in same_list:
        in_edges_z = get_in_path(z, promo_node, circuit_tf_list)
        for edge in in_edges_z:
            if edge not in edge_list:
                edge_list.append(edge)

        in_edges_i = get_in_path(i, promo_node, circuit_tf_list)
        for edge in in_edges_i:
            if edge not in edge_list:
                edge_list.append(edge)

        out_edges_z = get_out_path(z, part_list, circuit_tf_list)
        for edge in out_edges_z:
            if edge not in edge_list:
                edge_list.append(edge)

        all_out_edges_z = [edge for edge in (in_edges_z + in_edges_i)
                           if edge[0] == z] + out_edges_z
        inhibitor_edges = [(i, k[1]) for k in all_out_edges_z]

        for edge in inhibitor_edges:
            if edge not in edge_list:
                edge_list.append(edge)

    return edge_list


def get_dose(min_dose=10, max_dose=75, dose_interval=5, num_part=1):
    """Sample random plasmid doses (ng) uniformly on [min_dose, max_dose] at fixed intervals."""
    return (np.random.choice(np.arange(min_dose, max_dose + 1, dose_interval),
                             size=num_part, replace=True))


def sampling(promo_node, num_dict, min_dose, max_dose, dose_interval, inhibitor=False):
    """Generate an initial population of random circuit topologies."""
    combo = []
    for num_part, num_circuit in num_dict.items():
        num_part = int(num_part)

        if not inhibitor:
            part_combo = list(combinations(tf_list, num_part))
            ind = np.random.choice(len(part_combo), num_circuit)
            combo.extend([part_combo[i] for i in ind])
        else:
            for i in range(num_circuit):
                guaranteed_tf_index = np.random.choice(len(tf_list), 1)
                guaranteed_tf = tf_list[guaranteed_tf_index[0]]
                remaining_tfs = np.delete(tf_list, guaranteed_tf_index[0])
                remaining_options = np.concatenate((remaining_tfs, inhibitor_list))

                choices = np.random.choice(remaining_options, size=num_part - 1)
                combo.extend([np.append(choices, guaranteed_tf)])

    circuits = []
    for i in range(len(combo)):
        part_list = combo[i]
        edge_list = get_edges(promo_node, list(part_list))
        dose_list = dict(zip(part_list,
                             get_dose(min_dose, max_dose, dose_interval, len(part_list))))
        circuits.append([Topo(edge_list, dose_list, promo_node)])

    return np.asarray(circuits)


def validate(g):
    """Repair circuit connectivity in-place after a mutation to restore graph validity."""
    circuit_tf_list = []
    same_list = []

    for k in g.part_list:
        if k[0] == 'Z':
            circuit_tf_list.append(k)
            if ('I' + k[1:]) in g.part_list:
                same_list.append([k, ('I' + k[1:])])

    if len(circuit_tf_list) == 0:
        raise Exception("Something's wrong. No TFs in the circuit.")

    if (('Rep' not in g.graph.nodes) |
        (len([k for k in g.graph.predecessors('Rep') if k[0] == 'Z']) == 0)):
        g.graph.add_edges_from(get_in_path('Rep', None, circuit_tf_list))

    for n in g.part_list:
        if n not in g.graph.nodes:
            g.graph.add_edges_from(get_in_path(n, g.promo_node, circuit_tf_list))
            g.graph.add_edges_from(get_out_path(n, g.part_list, circuit_tf_list))

        else:
            viable_type = [k[-2][0]
                           for k in nx.all_simple_paths(g.graph, g.promo_node, n)]

            if len(viable_type) == 0:
                g.graph.add_edges_from(get_in_path(n, g.promo_node, circuit_tf_list))
            else:
                if (('I' in viable_type) and
                    (('Z' not in viable_type) and ('P' not in viable_type))):
                    g.graph.add_edges_from(get_in_path(n, g.promo_node, circuit_tf_list))

            if len(list(nx.all_simple_paths(g.graph, n, 'Rep'))) == 0:
                g.graph.add_edges_from(get_out_path(n, g.part_list, circuit_tf_list))

            predecessor_types = [k[0] for k in g.graph.predecessors(n)]
            if ("I" in predecessor_types) & ("Z" not in predecessor_types):
                z_reg = np.random.choice(circuit_tf_list)
                g.graph.add_edges_from([(z_reg, n)])

    if set(g.graph.edges) != set(g.edge_list):
        g.update(list(g.graph.edges))

    for z, i in same_list:
        z_succ = set(g.graph.successors(z))
        i_succ = set(g.graph.successors(i))
        if z_succ != i_succ:
            z_out = list(g.graph.out_edges(z))
            i_out = list(g.graph.out_edges(i))
            i_out_new = [(i, k[1]) for k in z_out]
            g.graph.remove_edges_from(i_out)
            g.graph.add_edges_from(i_out_new)

    if set(g.graph.edges) != set(g.edge_list):
        g.update(list(g.graph.edges))


def check_valid(g, promo_node, part_list):
    """Return 1 if the circuit graph satisfies all structural constraints, else 0."""
    if (promo_node not in g.nodes) or ('Rep' not in g.nodes):
        return 0

    circuit_tf_list = []
    same_list = []
    for k in part_list:
        if k[0] == 'Z':
            circuit_tf_list.append(k)
            if ('I' + k[1:]) in part_list:
                same_list.append([k, ('I' + k[1:])])

    if len(circuit_tf_list) == 0:
        return 0

    if (('Rep' not in g.nodes) |
        (len([k for k in g.predecessors('Rep') if k[0] == 'Z']) == 0)):
        return 0

    for n in part_list:
        if n not in g.nodes:
            return 0
        else:
            viable_type = [k[-2][0]
                           for k in nx.all_simple_paths(g, promo_node, n)]
            if len(viable_type) == 0:
                return 0
            else:
                if (('I' in viable_type) and
                    (('Z' not in viable_type) and ('P' not in viable_type))):
                    return 0
            if len(list(nx.all_simple_paths(g, n, 'Rep'))) == 0:
                return 0

            predecessor_types = [k[0] for k in g.predecessors(n)]
            if ("I" in predecessor_types) & ("Z" not in predecessor_types):
                return 0

    for z, i in same_list:
        z_succ = set(g.successors(z))
        i_succ = set(g.successors(i))
        if z_succ != i_succ:
            return 0
    return 1


def compare_circuit(g1, g2):
    """Return True if two circuits share identical edge sets and dose mappings."""
    ind = (set(g1.edge_list) == set(g2.edge_list)) & (g1.dose == g2.dose)
    return ind


def get_crosspt(list1, list2):
    """Find a compatible crossover part pair between two circuits' part lists."""
    same = []
    for item in list1:
        if item in list2:
            same.append(item)

    if len(same) > 0:
        pt1 = np.random.choice(same)
        pt2 = pt1
    else:
        pt1 = np.random.choice(list1)
        if pt1[0] == 'Z':
            pt2 = np.random.choice([k for k in list2 if k[0] == 'Z'])
        else:
            list2_inhibitors = [k for k in list2 if k[0] == 'I']
            if len(list2_inhibitors) > 0:
                pt2 = np.random.choice(list2_inhibitors)
            else:
                if len(list2) > 1:
                    pt2 = np.random.choice(list2)
                else:
                    pt1 = np.random.choice([k for k in list1 if k[0] == 'Z'])
                    pt2 = np.random.choice([k for k in list2 if k[0] == 'Z'])
    return pt1, pt2


def switch_node(g, old_node, new_node):
    """Rename old_node to new_node in all edges of the circuit graph."""
    child_edge = []
    for edge in list(g.graph.edges):
        source, target = edge
        if source == old_node:
            source = new_node
        if target == old_node:
            target = new_node
        edge = (source, target)
        child_edge.append(tuple(edge))
    return child_edge


def match_node(new_node, part_list, promo_node, circuit_tf_list,
               circuit_in_list, pt2, node_list2):
    """Map nodes from a donor sub-circuit into the recipient's compatible node set."""
    for n in node_list2:
        if n == pt2:
            new_node.append(pt2)
        elif n in (part_list + [promo_node, 'Rep']):
            new_node.append(n)
        elif n[0] == 'Z':
            node_avail = []
            for item in circuit_tf_list:
                if item not in new_node:
                    node_avail.append(item)
            if len(node_avail) > 0:
                n_new = np.random.choice(list(node_avail))
                new_node.append(n_new)
        elif n[0] == 'I':
            node_avail = []
            for item in circuit_in_list:
                if item not in new_node:
                    node_avail.append(item)
            if len(node_avail) > 0:
                n_new = np.random.choice(list(node_avail))
                new_node.append(n_new)


def switch_edge(g1, pt1, pt2, in_list2, out_list2, dose2):
    """Swap part pt1 in g1 for pt2 from g2, rewiring edges and validating the result."""
    child = deepcopy(g1)
    child.part_list.remove(pt1)
    child.dose.pop(pt1)
    child.graph.remove_node(pt1)

    circuit_tf_list = [k for k in child.part_list if k[0] == 'Z']
    circuit_in_list = [k for k in child.part_list if k[0] == 'I']

    in_node = []
    common_list2 = [k for k in in_list2 if k in out_list2]
    match_node(in_node, child.part_list, child.promo_node, circuit_tf_list, circuit_in_list, pt2, common_list2)
    out_node = [k for k in in_node if k[0] != 'P']

    in_list2 = [k for k in in_list2 if k not in common_list2]
    match_node(in_node, child.part_list, child.promo_node, circuit_tf_list, circuit_in_list, pt2, in_list2)
    out_list2 = [k for k in out_list2 if k not in common_list2]
    match_node(out_node, child.part_list, child.promo_node, circuit_tf_list, circuit_in_list, pt2, out_list2)

    new_edges = []
    for k in in_node:
        new_edges.append((k, pt2))
    for k in out_node:
        out_edge = (pt2,k)
        if out_edge not in new_edges:
            new_edges.append(out_edge)

    child.part_list.append(pt2)
    child.dose.update({pt2: dose2})
    child.graph.add_edges_from(new_edges)
    validate(child)
    return child


def crossover_structure(g1, g2):
    """Produce two offspring by swapping one compatible part between two parent circuits."""
    pt1, pt2 = get_crosspt(g1.part_list, g2.part_list)
    child1 = switch_edge(g1, pt1, pt2, list(g2.graph.predecessors(pt2)),
                          list(g2.graph.successors(pt2)), g2.dose[pt2])
    child2 = switch_edge(g2, pt2, pt1, list(g1.graph.predecessors(pt1)),
                         list(g1.graph.successors(pt1)), g1.dose[pt1])
    return child1, child2


def mutate_dose(g, min_dose=10, max_dose=75, dose_interval=5):
    """Randomly resample the plasmid dose for one part in the circuit."""
    n = np.random.choice(g.part_list)
    g.dose.update({n: get_dose(min_dose, max_dose, dose_interval, 1)[0]})


def mutate_node_type(g, min_dose=10, max_dose=75, dose_interval=5):
    """Swap one part to the opposite type (activator Z ↔ inhibitor I) at the same index."""
    old_node = np.random.choice(g.part_list)
    if old_node[0] == 'Z':
        same_type = 'I'
        node_avail = []
        for item in tf_list:
            if item not in g.part_list:
                node_avail.append(item)
    else:
        same_type = 'Z'
        node_avail = []
        for item in inhibitor_list:
            if item not in g.part_list:
                node_avail.append(item)

    if len(node_avail) > 0:
        new_node = np.random.choice(node_avail)
        same = same_type + new_node[1:]
        g.dose.update({new_node: g.dose[old_node]})
        g.dose.pop(old_node)
        new_edges = switch_node(g, old_node, new_node)
        g.graph.remove_node(old_node)
        g.update(new_edges)
        if same in g.part_list:
            same_out = list(g.graph.out_edges(same))
            new_node_out = list(g.graph.out_edges(new_node))
            if same_type == 'Z':
                new_node_out_new = [(new_node, k[1]) for k in same_out]
                g.graph.remove_edges_from(new_node_out)
                g.graph.add_edges_from(new_node_out_new)
            else:
                same_out_new = [(same, k[1]) for k in new_node_out]
                g.graph.remove_edges_from(same_out)
                g.graph.add_edges_from(same_out_new)
            g.update(list(g.graph.edges))


def add_node(g, circuit_tf_list, min_dose=10,
             max_dose=75, dose_interval=5, inhibitor=False):
    """Insert a new randomly wired part into the circuit and repair connectivity."""
    if not inhibitor:
        node_avail = [k for k in tf_list if k not in circuit_tf_list]
    else:
        node_avail = [k for k in parts.keys() if k not in g.part_list]

    new_node = np.random.choice(node_avail)
    new_node_type = new_node[0]
    if new_node_type == 'Z':
        circuit_tf_list.append(new_node)
    g.dose.update({new_node: get_dose(min_dose,
                                      max_dose, dose_interval, 1)[0]})

    new_edges = []
    for k in g.edge_list:
        new_edges.append(k)

    new_node_in = get_in_path(new_node, g.promo_node, circuit_tf_list)
    for edge in new_node_in:
        if edge not in new_edges:
            new_edges.append(edge)

    if new_node_type == 'I':
        same_type = 'Z'
        same = same_type + new_node[1:]
        if same in g.part_list:
            same_out = list(g.graph.out_edges(same))
            all_out_edges_same = [edge for edge in new_node_in if edge[0] == same] + same_out
            for k in all_out_edges_same:
                possible_edge = (new_node, k[1])
                if possible_edge not in new_edges:
                    new_edges.append(possible_edge)
        else:
            new_node_out = get_out_path(new_node, g.part_list, circuit_tf_list)
            for edge in new_node_out:
                if edge not in new_edges:
                    new_edges.append(edge)
    else:
        same_type = 'I'
        same = same_type + new_node[1:]
        if same in g.part_list:
            same_out = list(g.graph.out_edges(same))
            for i in range(len(new_edges)):
                if new_edges[i] in same_out:
                    new_edges.pop(i)

            new_node_out = get_out_path(new_node, g.part_list, circuit_tf_list)
            for edge in new_node_out:
                if edge not in new_edges:
                    new_edges.append(edge)

            all_out_edges_new = [edge for edge in new_node_in if edge[0] == new_node] + new_node_out
            for k in all_out_edges_new:
                possible_edge = (same, k[1])
                if possible_edge not in new_edges:
                    new_edges.append(possible_edge)
        else:
            new_node_out = get_out_path(new_node, g.part_list, circuit_tf_list)
            for edge in new_node_out:
                if edge not in new_edges:
                    new_edges.append(edge)

    g.update(new_edges)


def remove_node(g, circuit_tf_list):
    """Remove a random part from the circuit and repair connectivity."""
    circuit_in_list = [k for k in g.part_list if k[0] == 'I']

    if len(circuit_tf_list) > 1 | len(circuit_in_list) > 1:
        if len(circuit_in_list) <= 1:
            old_node = np.random.choice(circuit_tf_list)
        elif len(circuit_tf_list) <= 1:
            old_node = np.random.choice(circuit_in_list)
        else:
            old_node = np.random.choice(g.part_list)

        g.part_list.remove(old_node)
        g.dose.pop(old_node)
        g.graph.remove_node(old_node)
        validate(g)


def mutate_node_num(g, max_part, min_dose=10, max_dose=75,
                    dose_interval=5, inhibitor=False):
    """Add or remove one part depending on current circuit size vs. max_part."""
    circuit_tf_list = [k for k in g.part_list if k[0] == 'Z']
    if max_part > 1:
        if len(g.part_list) == 1:
            add_node(g, circuit_tf_list, min_dose, max_dose,
                     dose_interval, inhibitor)
        elif len(g.part_list) < max_part:
            if np.random.uniform() < 0.5:
                add_node(g, circuit_tf_list, min_dose, max_dose,
                         dose_interval, inhibitor)
            else:
                remove_node(g, circuit_tf_list)
        else:
            remove_node(g, circuit_tf_list)


def get_full_connected(part_list, promo_node):
    """Return the fully-connected edge list for a given part set and promoter."""
    edge_list = [(promo_node, k) for k in part_list]
    edge_list.extend([(k, 'Rep') for k in part_list])
    edge_list.extend([(k, k) for k in part_list])
    edge_list.extend(list(permutations(part_list, 2)))
    return edge_list


def mutate_edge(g, inhibitor=False):
    """Randomly add or remove one edge from the circuit while preserving graph validity."""
    if len(g.part_list) == 1:
        if g.graph.size() == 3:
            g.edge_list.remove((g.part_list[0], g.part_list[0]))
        else:
            g.edge_list.append((g.part_list[0], g.part_list[0]))
        g.update(g.edge_list)
    else:
        edge_full = get_full_connected(g.part_list, g.promo_node)
        if g.graph.size() == len(edge_full):
            if inhibitor:
                valid = 0
                while (valid == 0) and (len(edge_full) > 0):
                    g_graph = deepcopy(g.graph)
                    ind = np.random.choice(len(edge_full))
                    edge_removed = edge_full[ind]
                    g_graph.remove_edge(edge_removed[0], edge_removed[1])
                    valid = check_valid(g_graph, g.promo_node, g.part_list)
                    edge_full.remove(edge_removed)
                if valid == 1:
                    g.update(list(g_graph.edges))
            else:
                ind = np.random.choice(g.graph.size())
                g.edge_list.remove(edge_full[ind])
                g.update(g.edge_list)

        else:
            if np.random.uniform() < 0.5:
                edge_avail = [k for k in edge_full if k not in g.graph.edges]
                ind = np.random.choice(len(edge_avail))
                g.edge_list.append(edge_avail[ind])
                g.update(g.edge_list)
                validate(g)
            else:
                edge_avail = [k for k in g.edge_list]
                valid = 0
                while (valid == 0) and (len(edge_avail) > 0):
                    g_graph = deepcopy(g.graph)
                    ind = np.random.choice(len(edge_avail))
                    edge_removed = edge_avail[ind]
                    g_graph.remove_edge(edge_removed[0], edge_removed[1])
                    valid = check_valid(g_graph, g.promo_node, g.part_list)
                    edge_avail.remove(edge_removed)
                if valid == 1:
                    g.update(list(g_graph.edges))


def crossover(X, obj, rank_dict=None, **kwargs):
    """Tournament-select parent pairs and produce offspring via structural crossover."""
    n_matings = int(len(X) / 2)
    Y = np.full_like(X, None, dtype=object)
    for k in range(n_matings):
        throuple = np.random.choice(range(len(X)), 3, replace=False)
        if rank_dict is None:
            parents = throuple[obj[throuple].argsort()][:-1]
        else:
            throuple_rank = np.asarray([rank_dict[i]['rank'] for i in throuple])
            parents = throuple[throuple_rank.argsort()][:-1]
        Y[2 * k, 0], Y[2 * k + 1, 0] = crossover_structure(X[parents[0], 0], X[parents[1], 0])
    return Y


def mutate(problem, X, prob, dose=False, **kwargs):
    """Apply a random mutation operator to each individual in the population."""
    for i in range(len(X)):
        if np.random.uniform(0, 1) < prob:
            if dose:
                r = np.random.choice(4)
            else:
                r = np.random.choice(3)

            if r == 0:
                mutate_node_num(X[i, 0], problem.max_part, problem.min_dose, problem.max_dose, problem.dose_interval,
                                problem.inhibitor)
            elif r == 1:
                mutate_node_type(X[i, 0], problem.min_dose, problem.max_dose, problem.dose_interval)
            elif r == 2:
                mutate_edge(X[i, 0], problem.inhibitor)
            else:
                mutate_dose(X[i, 0], problem.min_dose, problem.max_dose, problem.dose_interval)
