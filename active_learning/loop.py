import numpy as np
import pickle
import copy
import os
import time
from datetime import datetime
from .utils import generate_dynamic_targets, generate_prior_ensemble
from .lab import VirtualLab
from .designer import ExperimentDesigner
from .learner import Learner

class ActiveLearningLoop:
    """
    The Master Controller for the Active Learning Pipeline.
    Manages the data flow between the Lab, Designer, and Learner.
    """
    def __init__(self, circuit_dict, true_parts, config):
        """Load data files, initialize all modules, and generate the prior belief cloud."""
        self.circuit_dict = circuit_dict
        self.true_parts = true_parts
        self.config = config
        
        # 1. Load Textbook Nominal Parts
        with open(self.config.nominal_parts_path, "rb") as f:
            self.nominal_parts = pickle.load(f)
            
        with open(self.config.promo_path, "rb") as f:
            self.promo_params = pickle.load(f)

        # 2. Generalization: Dynamically extract active parts & their parametric indices
        self.targets = generate_dynamic_targets(list(self.circuit_dict.values()))
        print(f"\n[AL Engine] Initialization Complete. Dynamic Targets Identified: {self.targets}")

        # 3. Initialize Modules
        self.lab = VirtualLab(self.circuit_dict, self.true_parts, self.promo_params, self.config)
        self.designer = ExperimentDesigner(self.circuit_dict, self.config)
        self.learner = Learner(self.circuit_dict, self.targets, self.config)
        
        # 4. Generate the Initial Prior Belief Cloud
        self.belief_cloud = generate_prior_ensemble(self.nominal_parts, self.targets, self.config)

    def run(self):
        """
        Executes the autonomous Active Learning Loop for 'max_cycles'.
        Returns the final posterior mean of the calibrated parameters.
        """
        history = []
        start_time = time.time()  # Start the timer!
        
        for cycle in range(self.config.max_cycles):
            print(f"\n" + "="*40)
            print(f" 🚀 STARTING AL CYCLE {cycle}")
            print("="*40)
            
            # Phase 1: Designer selects experiments
            selected_experiments, variance_matrix, all_simulations = self.designer.design_experiment(
                self.belief_cloud, self.promo_params
            )
            print(f"[Designer] Selected Experiments: {selected_experiments}")
            
            # Phase 2: Lab runs experiments and adds noise
            lab_data_dict = {}
            for c_name, dose in selected_experiments:
                t_span, y_noisy = self.lab.run_experiment(c_name, dose)
                lab_data_dict[(c_name, dose)] = y_noisy
                
            # Phase 3: Learner updates belief cloud
            self.belief_cloud, best_error = self.learner.update_belief(
                self.belief_cloud, self.promo_params, lab_data_dict
            )
            
            # Track History
            history.append({
                'cycle': cycle,
                'error': best_error,
                'variance_max': np.max(variance_matrix)
            })

        end_time = time.time()
        total_time_seconds = end_time - start_time

        print("\n🎉 ACTIVE LEARNING LOOP FINISHED.")
        
        final_params = self._extract_final_parameters()
        self._save_run_summary(final_params, total_time_seconds)
        
        return final_params, history

    def _extract_final_parameters(self):
        """
        Calculates the mean of the final belief cloud for the targeted parts
        and prints them side-by-side with the Hidden Truth.
        """
        final_params = {}
        for part in self.targets.keys():
            part_matrices = [model[part] for model in self.belief_cloud]
            final_params[part] = np.mean(part_matrices, axis=0)
            
        print("\n🎯 FINAL DISCOVERED PARAMETERS vs HIDDEN TRUTH:")
        for k, v in final_params.items():
            print(f"   {k}:")
            print(f"      AL Found: {np.round(v, 6)}")
            print(f"      Truth:    {np.round(self.true_parts[k], 6)}")
            
        return final_params

    def _save_run_summary(self, final_params, total_time_seconds):
        """
        Writes a human-readable .txt file logging the configurations and results.
        """
        os.makedirs("run_logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"run_logs/AL_Run_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write("========================================\n")
            f.write(" ACTIVE LEARNING LOOP - RUN SUMMARY\n")
            f.write("========================================\n\n")
            
            f.write(f"⏱️ Total Running Time: {total_time_seconds / 60:.2f} minutes\n\n")
            
            f.write("⚙️ CONFIGURATIONS:\n")
            f.write(f"   Max Cycles:        {self.config.max_cycles}\n")
            f.write(f"   Ensemble Size:     {self.config.ensemble_size}\n")
            f.write(f"   Prior Distribution:{self.config.dist_type}\n")
            f.write(f"   Budget Circuits:   {self.config.budget_circuits}\n")
            f.write(f"   Budget Dosages:    {self.config.budget_dosages}\n")
            f.write(f"   Candidate Dosages: {self.config.dosages}\n\n")
            
            f.write("🎯 DISCOVERED VS TRUTH:\n")
            for k, v in final_params.items():
                f.write(f"   Part: {k}\n")
                f.write(f"      AL Found: {np.round(v, 6)}\n")
                f.write(f"      Truth:    {np.round(self.true_parts[k], 6)}\n\n")
                
        print(f"\n📄 Run summary automatically saved to: {filename}")