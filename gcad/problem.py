# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks, peak_prominences
from scipy.stats.mstats import gmean

# Import the modified equations
from gcad.equations import (
    system_equations_pop,
    system_equations_DsRed_pop
)

# Import the DEFAULTS (for the Miner to use)
from gcad.loader import (
    Ref,
    Z_20,
    Ref_pop20,
    Z_200,
    Ref_pop200,
    promo,
    parts
)

# create dictionary of reference simulations
pop_model_ref = {"20 cell": Ref_pop20, "200 cell": Ref_pop200}

class PulseGenerator:
    """Multi-objective optimization problem for discovering pulse-generating genetic circuits."""

    def __init__(
            self,
            promo_node: str,
            dose_specs: list,
            max_part: int,
            inhibitor: bool,
            DsRed_inhibitor: bool,
            num_dict: dict, 
            n_gen: int,
            probability_crossover: float, 
            probability_mutation: float,
            mutate_dose: bool=False,
            pop: bool=False,
            mean: str="arithmetic",
            Z_mat: np.ndarray=Z_20,
            Ref_pop: dict=None, 
            num_processes: int=None,
            obj_labels: list=["t_pulse", "prominence_rel"],
            max_time: float=42,
            single_cell_tracking: bool=False
            ) -> None:
        """Configure the pulse-generator problem with GA settings and objective functions."""
        # Standard Attribute Assignment
        self.promo_node = promo_node
        self.min_dose = dose_specs[0]
        self.max_dose = dose_specs[1]
        self.dose_interval = dose_specs[2]
        self.max_part = max_part
        self.inhibitor = inhibitor
        self.num_dict = num_dict
        self.n_gen = n_gen
        self.prob_crossover = probability_crossover
        self.prob_mutation = probability_mutation
        self.mutate_dose = mutate_dose
        self.pop = pop
        self.mean = mean
        self.num_processes = num_processes
        self.obj_labels = obj_labels
        self.max_time = max_time
        
        # Select the Equation Function
        self.system_eqs = system_equations_pop
        if inhibitor and DsRed_inhibitor:
            self.system_eqs = system_equations_DsRed_pop

        if pop:
            self.Z = Z_mat
            if Ref_pop is not None:
                self.ref = Ref_pop
            else:
                self.ref = pop_model_ref[str(len(self.Z))+" cell"]
            self.simulate = self.simulate_pop
        else:
            self.ref = Ref
            self.Z = None
            self.simulate = self.simulate_cell

        # Select the Objective Function Logic (Simplified logic for brevity)
        if "t_pulse" in self.obj_labels and "prominence_rel" in self.obj_labels:
            self.func = self.func_obj_t_pulse
        elif "peak_rel" in self.obj_labels:
            self.func = self.func_obj_peak_rel
        else:
            self.func = self.func_obj_t_pulse # Default

    def simulate_cell(
        self,
        topology: object,
        Z_row: np.ndarray = np.ones(5)
    ):
        """Solves the ODEs for a given topology for a single cell."""
        
        max_time = self.max_time
        t = np.arange(0, max_time + 1, 1)
        
        # === THE CRITICAL MODIFICATION ===
        # We now pass the DEFAULT promo and parts dicts to the solver.
        # This keeps the 'Miner' working exactly as before.
        rep_on_ts = odeint(
            self.system_eqs,
            np.zeros(topology.num_states * 2),
            t, 
            args=('on', Z_row, topology, promo, parts) # <--- Passed Args Here
        )[:, -1]
        return t, rep_on_ts

    def simulate_pop(
        self, 
        topology: object, 
    ):
        """Solves the ODEs for a population."""
        rep_on_ts_all = []
        nc = len(self.Z)
        zipped_args = list(zip([topology]*nc, self.Z))
        
        for cell in range(0, nc):
            # Calls simulate_cell, which now handles the extra args
            t, rep_on_ts = self.simulate_cell(                
                zipped_args[cell][0],
                zipped_args[cell][1]
            )
            rep_on_ts_all.append(rep_on_ts)
            
        rep_on_ts_means = [np.mean(k) for k in zip(*rep_on_ts_all)]
        return t, rep_on_ts_means

    def calc_rep_rel(
        self,
        topology: object,
        rep_on_ts: list
    ):
        """Calculates relative reporter expression."""
        reference_on = self.ref[topology.promo_node]['on']
        rep_on_ts_rel = [i/reference_on for i in rep_on_ts]
        return rep_on_ts_rel

    @staticmethod
    def calc_peak_rel(rep_on_ts_rel: list):
        """Return the maximum relative reporter expression."""
        return max(rep_on_ts_rel)

    @staticmethod
    def calc_prominence_rel(rep_on_ts_rel: list, peak_rel: float):
        """Return the prominence of the first peak relative to the baseline."""
        peaks_rep, _ = find_peaks(
            rep_on_ts_rel,
            prominence=0.1*peak_rel
        )
        prominence_rep_list = peak_prominences(rep_on_ts_rel, peaks_rep)[0]
        if len(prominence_rep_list) == 0:
            prominence_rel = 0
        else:
            prominence_rel = prominence_rep_list[0]
        return prominence_rel
    
    def calc_t_pulse(
        self,
        t: np.ndarray,
        rep_on_ts_rel: list,
        peak_rel: float,
        prominence_rel: float
    ):
        """Return the time at which peak reporter expression occurs."""
        if prominence_rel != 0:
            idx_peak_rel = rep_on_ts_rel.index(peak_rel)
            t_pulse = t[idx_peak_rel]
        else:
            t_pulse = self.max_time
        return t_pulse

    def func_obj_t_pulse(
        self,
        topology: object
    ):
        """Standard Objective Function for Pulse Generator"""
        t, rep_on_ts = self.simulate(topology)
        rep_on_ts_rel = self.calc_rep_rel(topology, rep_on_ts)
        peak_rel = self.calc_peak_rel(rep_on_ts_rel)
        prominence_rel = self.calc_prominence_rel(rep_on_ts_rel, peak_rel)

        t_pulse = self.calc_t_pulse(
            t, rep_on_ts_rel, peak_rel, prominence_rel)
        
        # Removed some unused function for clearity

        # return negative prominence_rel for minimization
        return [t_pulse, -prominence_rel]