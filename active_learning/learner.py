import numpy as np
from scipy.integrate import odeint
from joblib import Parallel, delayed
import copy
from tqdm import tqdm
from gcad.equations import system_equations_DsRed_pop
from gcad.circuit import Topo

def _evaluate_single_model(p_mut, lab_data_dict, circuit_dict, t_span, promo_params, ref_scaling):
    """Score one parameter set against all lab observations using total NMSE."""
    total_nmse = 0.0
    for (c_name, dose), lab_y in lab_data_dict.items():
        topology = circuit_dict[c_name]
        sim_dose = {k: v * dose for k, v in topology.dose.items() if k != 'Rep'}
        if 'Rep' in topology.dose: 
            sim_dose['Rep'] = topology.dose['Rep']
        temp_topo = Topo(topology.edge_list, sim_dose, topology.promo_node)

        try:
            y_raw = odeint(
                system_equations_DsRed_pop,
                np.zeros(temp_topo.num_states * 2),
                t_span,
                args=('on', np.ones(5), temp_topo, promo_params, p_mut)
            )[:, -1]
            
            y_pred = y_raw / ref_scaling
            data_scale = max(np.max(lab_y), 1.0)
            nmse = np.mean(((y_pred - lab_y) / data_scale) ** 2)
            total_nmse += nmse
        except Exception:
            total_nmse += float('inf')
            
    return total_nmse

class Learner:
    """
    The Inference Engine. Uses Approximate Bayesian Computation (ABC)
    with fixed Gaussian noise to update the generalized belief ensemble.
    """

    def __init__(self, circuit_dict, targets, config):
        """Initialize with circuit dict, inference targets, and config."""
        self.circuit_dict = circuit_dict
        self.targets = targets
        self.config = config
        self.current_cycle = 0  # Track cycles for cooling schedule

    def update_belief(self, current_ensemble, promo_params, lab_data_dict):
        """Run one ABC selection-and-resample cycle; return updated ensemble and best NMSE."""
        t_span = self.config.get_t_span()
        print(f"\n[Learner] Assessing {len(current_ensemble)} models against {len(lab_data_dict)} experiments")

        # THE MULTIPROCESSING LOGIC
        errors = Parallel(n_jobs=-1)(
            delayed(_evaluate_single_model)(
                p_mut, lab_data_dict, self.circuit_dict, t_span, promo_params, self.config.reference_scaling
            ) for p_mut in current_ensemble
        )
        errors = np.array(errors)

        cutoff_count = max(int(len(current_ensemble) * self.config.selection_ratio), 5) 
        best_indices = np.argsort(errors)[:cutoff_count]
        
        survivors = [current_ensemble[i] for i in best_indices]
        best_error = errors[best_indices[0]]
        
        print(f"[Learner] Selected top {len(survivors)} candidates.")
        print(f"[Learner] Best Total NMSE: {best_error:.4e}")
        
        new_ensemble = []
        target_size = len(current_ensemble)

        while len(new_ensemble) < target_size:
            parent_idx = np.random.randint(len(survivors))
            child = copy.deepcopy(survivors[parent_idx])
            self._perturb_parameters(child)
            new_ensemble.append(child)
            
        self.current_cycle += 1 

        return new_ensemble, best_error

    def _perturb_parameters(self, param_dict):
        """Apply exponentially-decaying multiplicative Gaussian noise to targeted parameters."""
        # Perturbation scale shrinks each cycle to fine-tune rather than explore broadly
        decayed_scale = self.config.perturbation_scale * np.exp(-self.current_cycle / 2.0)
        
        for part, indices in self.targets.items():
            if part in param_dict:
                for idx in indices:
                    # Use the shrinking scale instead of the fixed scale for perturbation
                    noise_factor = np.random.normal(1.0, decayed_scale)
                    param_dict[part][idx] *= noise_factor