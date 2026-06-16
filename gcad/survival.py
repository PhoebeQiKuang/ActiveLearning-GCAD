# Adopted from GCAD Repo
# Edited by Phoebe Kuang

"""
rankcrowding.py (Merged with metrics)
Implements Rank and Crowding Survival for the Genetic Algorithm.
Adapted from pymoo to stand alone.
"""

import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival

def calc_crowding_distance(F, filter_out_duplicates=True, epsilon=1e-5):
    """
    Calculates the Crowding Distance for a set of objective values F.
    High crowding distance = Good (means the solution is unique/isolated).
    """
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    # Sort each objective and calculate distances
    dist = np.zeros(n_points)
    
    # Check for duplicates (optional stability check)
    if filter_out_duplicates:
        # Simple check: if rows are identical, distance is 0
        _, unique_indices = np.unique(F, axis=0, return_index=True)
        is_duplicate = np.ones(n_points, dtype=bool)
        is_duplicate[unique_indices] = False
    else:
        is_duplicate = np.zeros(n_points, dtype=bool)

    for i in range(n_obj):
        # Sort indices for this objective
        I = np.argsort(F[:, i])
        
        # Norm: Range of this objective
        norm = F[I[-1], i] - F[I[0], i]
        if norm == 0:
            norm = epsilon # Prevent divide by zero

        # Boundary points get infinity (always keep them)
        dist[I[0]] = np.inf
        dist[I[-1]] = np.inf

        # For points in between, distance is diff between neighbors
        # d[i] = (f[i+1] - f[i-1]) / (f_max - f_min)
        dist[I[1:-1]] += (F[I[2:], i] - F[I[:-2], i]) / norm

    # Duplicates get 0 distance (so they are removed first)
    dist[is_duplicate] = 0.0
    
    return dist

class RankAndCrowding(Survival):
    """
    Survival Strategy for NSGA-II.
    1. Sort population into Non-Dominated Fronts (Rank).
    2. If the last allowed front is too big, sort by Crowding Distance.
    """
    def __init__(self, nds=None):
        """Initialize with an optional non-dominated sorting strategy (defaults to pymoo NDS)."""
        super().__init__(filter_infeasible=True)
        # Use pymoo's efficient C-based sorting if available
        self.nds = nds if nds is not None else NonDominatedSorting()

    def do(self, F, n_survive=None, return_rank=False):
        """Select n_survive individuals by Pareto rank then crowding distance."""
        survivors = []

        # 1. Non-Dominated Sorting (Rank)
        # Returns list of arrays: [[indices of front 1], [indices of front 2], ...]
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        rank_dict = dict()

        for k, front in enumerate(fronts):
            # Calculate crowding distance for everyone in this front
            # F[front, :] selects the objective values for these individuals
            crowding_of_front = calc_crowding_distance(F[front, :])

            # Case A: The whole front fits into the survivor list
            if len(survivors) + len(front) <= n_survive:
                for j, i in enumerate(front):
                    rank_dict.update({i: {"rank": k, "crowding": crowding_of_front[j]}})
                survivors.extend(front)

            # Case B: The front is too big. We must pick the best ones.
            else:
                n_needed = n_survive - len(survivors)
                
                # Sort by Crowding Distance DESCENDING (Largest distance = Most unique = Best)
                # randomized_argsort handles ties randomly to prevent bias
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:n_needed]

                # Add the winners
                for j in I:
                    idx = front[j]
                    rank_dict.update({idx: {"rank": k, "crowding": crowding_of_front[j]}})
                
                survivors.extend(front[I])

        if return_rank:
            return survivors, rank_dict
        else:
            return survivors