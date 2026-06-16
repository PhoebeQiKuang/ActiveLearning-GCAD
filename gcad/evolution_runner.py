# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import numpy as np
import pickle
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from gcad.survival import RankAndCrowding
from gcad.evolution import crossover, mutate


def _eval_circuit(circuit, problem):
    """Evaluate the fitness objectives for a single circuit topology."""
    return problem.func(circuit)


def multi_obj_GA(
        folder_path: str,
        problem: object, 
        population: np.ndarray,
        num_circuits: int,
        obj: np.ndarray,
        get_unique: bool=False,
        plot: bool=False
):
    """
    Runs the Genetic Algorithm with a TQDM progress bar.
    """
    
    # History storage
    all_obj = [obj]
    all_circuits = [population]

    # Survival Strategy (Sorting)
    nds = RankAndCrowding()
    
    print(f"Starting Evolution ({problem.n_gen} Generations)...")
    
    # This replaces 'alive_bar'
    iterator = tqdm(range(problem.n_gen), desc="Evolving", unit="gen")
    
    for gen in iterator:
        # 1. Selection (Sort Parent Population)
        _, rank_dict = nds.do(obj, num_circuits, return_rank=True)

        # 2. Crossover (Mating)
        if np.random.uniform() < problem.prob_crossover:
            children = crossover(population, obj, rank_dict)
        else:
            children = deepcopy(population)

        # 3. Mutation (Random Changes)
        mutate(problem, children, 
                problem.prob_mutation, 
                dose=problem.mutate_dose
        )

        # 4. Evaluation (Physics Simulation) — parallelized across all children
        obj_children = np.asarray(
            Parallel(n_jobs=-1)(delayed(_eval_circuit)(g[0], problem) for g in children)
        )
            
        # Store History
        all_obj.append(obj_children)
        all_circuits.append(children)

        # 5. Merge & Survive
        obj = np.vstack((obj, obj_children))
        population = np.vstack((population, children))

        # Keep only the best 'num_circuits'
        S = nds.do(obj, num_circuits)
        obj = obj[S]
        population = population[S, :]
        
        # Update progress bar description with best Prominence found so far
        # (Assuming 2nd objective is -Prominence, so min(obj) is max prominence)
        best_prom = -np.min(obj[:, 1]) 
        iterator.set_postfix({"Max Prominence": f"{best_prom:.2f}"})

    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\nSaving results to pickle files...")
    
    obj_df = pd.DataFrame(obj, columns=problem.obj_labels)

    # 1. The Winners (Final Generation)
    obj_df.to_pickle(f"{folder_path}/final_objectives_df.pkl")
    with open(f"{folder_path}/final_population.pkl", "wb") as fid:
        pickle.dump(population, fid)

    # 2. Unique Circuits (The Search Space)
    # Flatten history
    flat_circuits = np.vstack(all_circuits)
    flat_obj = np.vstack(all_obj)
    
    # De-duplicate logic
    circuit_edge_lists = []
    for circuit in flat_circuits:
        c = circuit[0]
        # Create a unique string signature: Edges + Doses
        sig = str(sorted(c.edge_list)) + str(sorted(c.dose.items()))
        circuit_edge_lists.append(sig)

    seen = set()
    unique_indices = []
    for i, sig in enumerate(circuit_edge_lists):
        if sig not in seen:
            seen.add(sig)
            unique_indices.append(i)

    unique_obj = flat_obj[unique_indices]
    unique_circuits = flat_circuits[unique_indices]
    
    unique_obj_df = pd.DataFrame(unique_obj, columns=problem.obj_labels)

    with open(f"{folder_path}/unique_objectives_df.pkl", "wb") as fid:
        pickle.dump(unique_obj_df, fid)

    with open(f"{folder_path}/unique_circuits.pkl", "wb") as fid:
        pickle.dump(unique_circuits, fid)
        
    print(f"Done! Results saved in {folder_path}")