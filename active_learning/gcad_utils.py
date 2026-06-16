import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Path to gcad/data/ resolved relative to this file — works regardless of working directory
_GCAD_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gcad', 'data'))

# ==========================================
# 1. THE GCAD WRAPPER
# ==========================================
def run_gcad_miner(settings_path, output_folder):
    """Runs the multi-objective GA using the current gcad/data/parts.pkl."""
    from gcad.evolution import sampling
    from gcad.evolution_runner import multi_obj_GA
    from gcad.problem import PulseGenerator

    with open(settings_path, "r") as f:
        settings = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    problem = PulseGenerator(
        promo_node=settings["promo_node"],
        dose_specs=settings["dose_specs"],
        max_part=settings["max_part"],
        inhibitor=settings["inhibitor"],
        DsRed_inhibitor=settings["DsRed_inhibitor"],
        num_dict=settings["num_dict"],
        n_gen=settings["n_gen"],
        probability_crossover=settings["probability_crossover"],
        probability_mutation=settings["probability_mutation"],
        mutate_dose=settings["mutate_dose"],
        pop=settings["pop"],
        obj_labels=settings["obj_labels"],
        max_time=settings["max_time"]
    )

    print(f"\n[GCAD] >> Running Evolution ({settings['n_gen']} Gen)...")
    population = sampling(
        problem.promo_node, problem.num_dict,
        problem.min_dose, problem.max_dose,
        problem.dose_interval, problem.inhibitor
    )

    obj = np.asarray(
        Parallel(n_jobs=-1)(delayed(problem.func)(ind[0]) for ind in population)
    )
    num_circuits = int(sum(settings["num_dict"].values()))

    multi_obj_GA(
        folder_path=output_folder,
        problem=problem,
        population=population,
        num_circuits=num_circuits,
        obj=obj,
        get_unique=False,
        plot=False
    )
    print(f"[GCAD] >> Evolution Saved to {output_folder}")

# ==========================================
# 2. THE CIRCUIT EXTRACTOR
# ==========================================
def abstract_edges(edge_list):
    """Convert a circuit edge list to a generic topology signature for architecture deduplication."""
    generic = []
    for u, v in edge_list:
        src = 'P1' if u == 'P1' else ('A' if u.startswith('Z') else ('R' if u.startswith('I') else u))
        tgt = 'Rep' if v == 'Rep' else ('A' if v.startswith('Z') else ('R' if v.startswith('I') else v))
        generic.append((src, tgt))
    return tuple(sorted(generic))

def extract_top_circuits(gcad_output_folder, m_target=4):
    """Parses the Pareto front for the top-performing circuits with different architectures."""
    print(f"\n[GCAD] >> Extracting Top {m_target} Architectures...")
    df_path = os.path.join(gcad_output_folder, "final_objectives_df.pkl")
    pop_path = os.path.join(gcad_output_folder, "final_population.pkl")

    df = pd.read_pickle(df_path)
    with open(pop_path, "rb") as f:
        pop = pickle.load(f)

    df['circuit_object'] = [ind[0] for ind in pop]
    if 'prominence_rel' in df.columns and df['prominence_rel'].mean() < 0:
        df['prominence_rel'] = -df['prominence_rel']

    df['abstract_sig'] = df['circuit_object'].apply(lambda c: abstract_edges(c.edge_list))

    valid_df = df[df['prominence_rel'] > 0.1].copy()
    if len(valid_df) == 0:
        print("[GCAD] WARNING: No valid pulses found. Using raw Top circuits.")
        valid_df = df.copy()

    abs_counts = valid_df['abstract_sig'].value_counts().reset_index()
    abs_counts.columns = ['abstract_sig', 'Count']
    top_archs = abs_counts.head(m_target)

    selected_circuits = {}
    for i, row in top_archs.iterrows():
        arch_sig = row['abstract_sig']
        subset = valid_df[valid_df['abstract_sig'] == arch_sig]

        if 'prominence_rel' in subset.columns:
            best_circuit_row = subset.sort_values('prominence_rel', ascending=False).iloc[0]
        else:
            best_circuit_row = subset.iloc[0]

        circ_obj = best_circuit_row['circuit_object']
        c_name = f"Circuit_{i+1}"
        selected_circuits[c_name] = circ_obj

    return selected_circuits

# ==========================================
# 3. THE LIBRARY UPDATER (With Tracking)
# ==========================================
def update_gcad_library(calibrated_parameters):
    """Safely overwrites gcad/data/parts.pkl and logs the parameter shift."""
    parts_path = os.path.join(_GCAD_DATA_DIR, 'parts.pkl')
    backup_path = os.path.join(_GCAD_DATA_DIR, 'parts_nominal_backup.pkl')

    if not os.path.exists(backup_path):
        shutil.copy(parts_path, backup_path)
        print(f"\n[GCAD] Created backup: parts_nominal_backup.pkl")

    with open(parts_path, 'rb') as f:
        parts_dict = pickle.load(f)

    print(f"\n[GCAD] >> UPDATING GCAD LIBRARY:")
    for part, new_vals in calibrated_parameters.items():
        old_vals = parts_dict[part]
        print(f"   PART: {part}")
        for idx in range(len(new_vals)):
            old = old_vals[idx]
            new = new_vals[idx]
            if old != new:
                pct_change = ((new - old) / old) * 100
                print(f"      Idx [{idx}]: {old:.4f}  -->  {new:.4f}  ({pct_change:+.2f}%)")
        parts_dict[part] = new_vals

    with open(parts_path, 'wb') as f:
        pickle.dump(parts_dict, f)

    print(f"[GCAD] Library updated. Ready for next cycle.")
