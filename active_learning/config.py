from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class ActiveLearningConfig:
    """
    User-facing configuration for the Active Learning Loop.
    Parameters can be editted to control the entire pipeline.
    """
    
    # 0. GCAD Settings
    nominal_parts_path: str = "gcad/data/parts.pkl"
    promo_path: str = "gcad/data/promo.pkl"
    
    # 1. Time array generation: [start, stop, num_steps]
    t_span_params: tuple = (0, 126, 126)

    # 2. Prior Belief Generation
    ensemble_size: int = 200
    spread_factor: float = 2.0
    dist_type: str = 'lognormal'  # Options: 'lognormal', 'uniform', 'u-shaped'

    # 3. Experimental Design
    # Default is 20 dosages from 0.2 to 4.0
    dosages: List[float] = field(default_factory=lambda: [round(0.2 * i, 2) for i in range(1, 21)]) 
    budget_circuits: int = 2      # How many circuits to pick per cycle
    budget_dosages: int = 2       # How many dosages to pick per circuit

    # 4. Lab Settings & Dimensional Consistency
    measurement_noise_std: float = 5.0
    reference_scaling: float = 66.60528765956212  # STRICT SCALING FACTOR

    # 5. Learner Settings (ABC)
    selection_ratio: float = 0.2
    perturbation_scale: float = 0.1   # Exponential decay starting point
    max_cycles: int = 10              # Strict stopping condition
    
    def get_t_span(self):
        """Helper to generate the actual numpy time array."""
        return np.linspace(self.t_span_params[0], self.t_span_params[1], self.t_span_params[2])