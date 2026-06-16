import numpy as np
import copy

def generate_dynamic_targets(circuit_list):
    """
    Read in a list of GCAD Topo objects (selected circuits), extracts unique biological parts,
    and maps them to their parametric indices based on the parts library structure.
    """
    active_parts = set()
    for circuit in circuit_list:
        active_parts.update(circuit.part_list)
        
    targets = {}
    for part in active_parts:
        if part.startswith('Z'):
            targets[part] = [0, 1, 2] # Activators: 3 parameters
        elif part.startswith('I') or part.startswith('R'):
            targets[part] = [0]       # Repressors: 1 parameter
        else:
            print(f"Warning: Unknown part '{part}'. Skipping.")
            
    return targets

def generate_prior_ensemble(nominal_parts, targets, config):
    """
    Generates the parameter belief cloud using the user's config settings.
    """
    ensemble = []
    low_bound = 1.0 / config.spread_factor
    high_bound = config.spread_factor
    sigma = np.log(config.spread_factor) / 2.0 
    
    for _ in range(config.ensemble_size):
        model_copy = copy.deepcopy(nominal_parts)
        
        for part, indices in targets.items():
            if part in model_copy:
                for idx in indices:
                    if config.dist_type == 'lognormal':
                        noise_factor = np.random.lognormal(mean=0.0, sigma=sigma)
                    elif config.dist_type == 'uniform':
                        noise_factor = np.random.uniform(low=low_bound, high=high_bound)
                    elif config.dist_type == 'u-shaped':
                        beta_sample = np.random.beta(a=0.5, b=0.5)
                        noise_factor = low_bound + beta_sample * (high_bound - low_bound)
                    else:
                        raise ValueError(f"Unknown distribution: {config.dist_type}")
                        
                    model_copy[part][idx] *= noise_factor
                    
        ensemble.append(model_copy)
        
    return ensemble