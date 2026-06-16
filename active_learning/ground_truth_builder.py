import copy
import pickle

def generate_synthetic_truth(nominal_parts_path):
    """
    Creates a 'Hidden Reality' by mathematically mutating the nominal parameters.
    This simulates a scenario where the parameters are universally flawed across all dimensions.
    """
    with open(nominal_parts_path, "rb") as f:
        textbook_parts = pickle.load(f)

    hidden_truth = copy.deepcopy(textbook_parts)

    for part, params in hidden_truth.items():
        if part.startswith('Z'):
            # Activators: Mutate all 3 dimensions
            # Example: Z[0] decreases by 25%, Z[1] increases by 25%, Z[2] increases by 25%
            hidden_truth[part][0] *= 1.25  
            hidden_truth[part][1] *= 0.75  
            hidden_truth[part][2] *= 0.75  
            
        elif part.startswith('I') or part.startswith('R'):
            # Repressors: Mutate the 1 dimension
            # Example: I[0] increases by 25%
            hidden_truth[part][0] *= 1.25

    return hidden_truth