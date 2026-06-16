import numpy as np
from scipy.integrate import odeint
from gcad.equations import system_equations_DsRed_pop
from gcad.circuit import Topo

class VirtualLab:
    """
    The Ground Truth Multi-Circuit Lab.
    Holds the UNIVERSAL 'true' parameters and generates noisy data for specific circuits.
    """
    def __init__(self, circuit_dict, true_universal_params, true_promo_params, config):
        """Initialize the lab with circuits, ground-truth parameters, and config."""
        self.circuits = circuit_dict
        self.true_universal_params = true_universal_params
        self.true_promo_params = true_promo_params
        self.config = config

    def run_experiment(self, circuit_id, dosage_factor):
        """Simulate a (circuit, dosage) experiment and return noisy time-series measurements."""
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit {circuit_id} not found in the lab.")
            
        topology = self.circuits[circuit_id]
        t_span = self.config.get_t_span()
            
        experimental_dose = {k: v * dosage_factor for k, v in topology.dose.items() if k != 'Rep'}
        if 'Rep' in topology.dose:
            experimental_dose['Rep'] = topology.dose['Rep']
        
        lab_bench_topology = Topo(topology.edge_list, experimental_dose, topology.promo_node)
        
        y_raw = odeint(
            system_equations_DsRed_pop,
            np.zeros(lab_bench_topology.num_states * 2),
            t_span,
            args=('on', np.ones(5), lab_bench_topology, self.true_promo_params, self.true_universal_params)
        )[:, -1]
        
        # STRICT SCALING
        y_measured = y_raw / self.config.reference_scaling
        
        # DYNAMIC NOISE (Config passes std as a percentage, e.g. 5.0 for 5%)
        noise_fraction = self.config.measurement_noise_std / 100.0
        noise = np.random.normal(0, noise_fraction * np.max(y_measured), size=len(y_measured))
        y_noisy = np.maximum(y_measured + noise, 0)
        
        return t_span, y_noisy