import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
from joblib import Parallel, delayed
from gcad.equations import system_equations_DsRed_pop
from gcad.circuit import Topo

def _simulate_designer_trace(param_dict, t_span, temp_topo, promo_params, ref_scaling):
    """Simulate one ODE trajectory for a given parameter set and topology."""
    try:
        y_raw = odeint(
            system_equations_DsRed_pop,
            np.zeros(temp_topo.num_states * 2),
            t_span,
            args=('on', np.ones(5), temp_topo, promo_params, param_dict)
        )[:, -1]
        return y_raw / ref_scaling
    except Exception:
        return np.zeros(len(t_span))

class ExperimentDesigner:
    """
    Calculates uncertainty across the belief cloud to select the most informative 
    (Circuit, Dosage) combinations based on the user's budget.
    """
    def __init__(self, circuit_dict, config):
        """Set up the designer with a circuit dictionary and config."""
        self.circuit_dict = circuit_dict
        self.config = config
        
        self.candidate_dosages = self.config.dosages
        self.p_circuits = min(self.config.budget_circuits, len(circuit_dict))
        self.q_dosages = min(self.config.budget_dosages, len(self.candidate_dosages))

    def design_experiment(self, belief_cloud, promo_params):
        """Simulate the full (circuit x dosage) grid and return the highest-variance experiments."""
        t_span = self.config.get_t_span()
        circuit_names = list(self.circuit_dict.keys())
        
        num_circuits = len(circuit_names)
        num_dosages = len(self.candidate_dosages)
        
        variance_matrix = np.zeros((num_circuits, num_dosages))
        all_simulations = {} 
        
        total_tasks = num_circuits * num_dosages
        with tqdm(total=total_tasks, desc="[Designer] Simulating Grid", unit="exp") as pbar:
            for i, c_name in enumerate(circuit_names):
                topology = self.circuit_dict[c_name]
                
                for j, dose in enumerate(self.candidate_dosages):
                    current_dose_map = {k: v * dose for k, v in topology.dose.items() if k != 'Rep'}
                    if 'Rep' in topology.dose: 
                        current_dose_map['Rep'] = topology.dose['Rep']
                        
                    temp_topo = Topo(topology.edge_list, current_dose_map, topology.promo_node)

                    # THE MULTIPROCESSING LOGIC
                    traces = Parallel(n_jobs=-1)(
                        delayed(_simulate_designer_trace)(
                            param, t_span, temp_topo, promo_params, self.config.reference_scaling
                        ) for param in belief_cloud
                    )
                    
                    all_simulations[(c_name, dose)] = traces 
                    
                    v_t = np.var(traces, axis=0)
                    variance_matrix[i, j] = np.mean(v_t)
                    
                    pbar.update(1) 

        selected_experiments = []
        row_totals = np.sum(variance_matrix, axis=1)
        top_p_circuit_indices = np.argsort(row_totals)[-self.p_circuits:][::-1]
        
        for c_idx in top_p_circuit_indices:
            c_name = circuit_names[c_idx]
            row_variances = variance_matrix[c_idx, :]
            top_q_dose_indices = np.argsort(row_variances)[-self.q_dosages:][::-1]
            
            for d_idx in top_q_dose_indices:
                dose = self.candidate_dosages[d_idx]
                selected_experiments.append((c_name, dose))
                
        return selected_experiments, variance_matrix, all_simulations