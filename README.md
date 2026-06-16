# Active Learning GCAD

Winter 2026 Rotation Project @ Mangan Group (Northwestern University)

An autonomous parameter inference framework for gene circuit design. The system uses **Optimal Experimental Design (OED)** and **Approximate Bayesian Computation (ABC)** to calibrate kinetic parameters against wet-lab measurements, then feeds the calibrated model back into the **GCAD** for more physics-accurate circuit discovery.

---

## Overview

Real biological parts deviate from textbook values. This framework closes the loop:

```
GCAD mines circuits  →  Active Learning calibrates parameters  →  GCAD re-mines with better model
```

**GCAD** runs a multi-objective genetic algorithm (NSGA-II) to discover gene circuit topologies that produce target dynamics (amplification, pulse, etc.). The resulting Pareto-optimal circuits are then used as experimental handles to infer the true kinetic parameters of the physical biological parts.

**Active Learning** iteratively selects the most informative (circuit, dosage) experiments using variance-based uncertainty sampling, runs them through a virtual lab (or real lab), and updates a parameter ensemble via ABC. After convergence, the calibrated parameters are written back to the GCAD parts library.

---

## Repository Structure

```
ActiveLearning-GCAD/
├── gcad/                          # GCAD circuit mining engine
│   ├── circuit.py                 # Topo: circuit topology object
│   ├── equations.py               # ODE system for DsRed population dynamics
│   ├── evolution.py               # Sampling, crossover, mutation operators
│   ├── evolution_runner.py        # NSGA-II main loop (parallelized)
│   ├── problem.py                 # PulseGenerator objective function
│   ├── survival.py                # Rank-and-crowding selection
│   ├── loader.py                  # Loads bundled parts/promo/ref data
│   ├── io.py                      # Results I/O utilities
│   └── data/                      # Bundled kinetic parameter libraries
│       ├── parts.pkl              # Kinetic parameters for all biological parts
│       ├── promo.pkl              # Promoter parameters
│       └── Ref.pkl                # Reference fluorescence values
│
├── active_learning/               # Active Learning inference engine
│   ├── config.py                  # ActiveLearningConfig (all tunable settings)
│   ├── loop.py                    # ActiveLearningLoop (top-level controller)
│   ├── lab.py                     # VirtualLab (noisy ODE simulator)
│   ├── designer.py                # ExperimentDesigner (OED via variance sampling)
│   ├── learner.py                 # Learner (ABC selection + resampling)
│   ├── utils.py                   # Prior generation, target detection
│   ├── gcad_utils.py              # Wrappers for GCAD mining + library update
│   └── ground_truth_builder.py    # Utility to generate synthetic ground truths
│
├── data/                          # Seed data committed to the repo
│   ├── gcad_results/              # GCAD Pareto front output + settings JSON
│   ├── ground_truth/              # Synthetic hidden truth (true_parts.pkl)
│   └── selected_M_circuits.pkl    # Top circuits extracted from gcad_results
│
├── workspace/                     # AL pipeline runtime files
│   └── prior_belief_cloud_*.pkl   # Pre-generated prior ensemble
│   (calibrated_parts.pkl and other outputs are written here at runtime)
│
├── results/                       # GCAD re-run outputs (gitignored)
│
├── notebooks/                     # Step-by-step workflow (run in order)
│   ├── 01_extract_circuits.ipynb
│   ├── 02_define_ground_truth.ipynb
│   ├── 03_generate_prior.ipynb
│   ├── 04_design_experiments.ipynb
│   ├── 05_run_active_learning.ipynb
│   ├── 06_gcad_rerun.ipynb
│   ├── 07_run_api.ipynb
│   └── 08_full_loop.ipynb
│
├── single_circuit_example/        # Single-circuit reference example
│   └── single_circuit_al.ipynb
│
├── define_circuit.py              # Pickle compatibility shim (do not remove)
├── example_walkthrough.ipynb      # Condensed end-to-end demo (START HERE)
├── setup.py
└── environment.yml
```

---

## Environment Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate GCAD_env
```

### 2. Install the packages (editable mode)

From the repo root:

```bash
pip install -e .
```

This registers both `gcad` and `active_learning` as importable packages from any working directory, so the notebooks work regardless of where Jupyter is launched from.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | Runtime |
| numpy | 1.26.4 | Numerical arrays |
| scipy | 1.11.4 | ODE integration (`odeint`) |
| pandas | 2.2.1 | Results DataFrames |
| matplotlib | 3.8 | Plotting |
| networkx | 3.1 | Circuit graph visualization |
| joblib | (via scipy) | Parallelized ODE evaluation |
| tqdm | (bundled) | Progress bars |

---

## Modules

### `gcad/`

Adapted from GCAD repo. Given a settings JSON, runs a multi-objective genetic algorithm to discover gene circuit topologies on the Pareto front of (time-to-pulse, prominence).

**Key classes and functions:**

- `gcad.circuit.Topo` — Represents a gene circuit. Stores `edge_list`, `dose`, `promo_node`, and derived state count. Required in scope when unpickling GCAD results.
- `gcad.equations.system_equations_DsRed_pop` — ODE right-hand side for the DsRed sequestration kinetics model.
- `gcad.evolution` — Sampling, crossover, and mutation operators for the GA.
- `gcad.evolution_runner.multi_obj_GA` — NSGA-II main loop. Accepts a `PulseGenerator` problem and runs for `n_gen` generations with parallelized fitness evaluation.
- `gcad.problem.PulseGenerator` — Multi-objective fitness function. Simulates a circuit and returns `[t_pulse, -prominence_rel]`.
- `gcad.loader` — Loads `parts`, `promo`, and reference data from `gcad/data/` using file-relative paths (works from any working directory).

GCAD settings are stored in `gcad/configs/FreeSearch_settings.json`. Copy this file to your working directory as `gcad_inputs/FreeSearch_settings.json` and edit it to change the search space.

---

### `active_learning/`

Bayesion parameter inference. Wraps the Lab, Designer, and Learner into an autonomous loop.

**`ActiveLearningConfig`** (`config.py`) — Single dataclass controlling the entire pipeline:

```python
from active_learning.config import ActiveLearningConfig

config = ActiveLearningConfig(
    max_cycles=10,           # AL iterations
    ensemble_size=200,       # number of parameter hypotheses
    spread_factor=2.0,       # prior width (parameters sampled ±log(2.0))
    dist_type='lognormal',   # prior distribution: 'lognormal', 'uniform', 'u-shaped'
    budget_circuits=2,       # circuits selected per cycle
    budget_dosages=2,        # dosages selected per circuit
    dosages=[...],           # candidate dosage scaling factors
    measurement_noise_std=5.0,   # lab noise (% of signal)
    selection_ratio=0.2,     # ABC: keep top 20% of models
    perturbation_scale=0.1,  # ABC: initial mutation scale (exponentially decayed)
)
```

**`ActiveLearningLoop`** (`loop.py`) — Top-level controller:

```python
from active_learning.loop import ActiveLearningLoop

engine = ActiveLearningLoop(circuit_dict, true_parts, config)
calibrated_parts, history = engine.run()
```

- `circuit_dict`: `dict[str, Topo]` — named circuits to use as experimental handles
- `true_parts`: `dict[str, np.ndarray]` — the hidden ground truth kinetic parameters
- Returns the posterior mean parameter dict and per-cycle history

**`VirtualLab`** (`lab.py`) — Simulates experiments with measurement noise:

```python
from active_learning.lab import VirtualLab

lab = VirtualLab(circuit_dict, true_parts, promo_params, config)
t_span, y_noisy = lab.run_experiment("Circuit_1", dosage_factor=1.4)
```

**`ExperimentDesigner`** (`designer.py`) — Variance-based OED across the (circuit × dosage) grid:

```python
from active_learning.designer import ExperimentDesigner

designer = ExperimentDesigner(circuit_dict, config)
selected_experiments, variance_matrix, all_simulations = designer.design_experiment(belief_cloud, promo_params)
```

Returns the top-budget (circuit, dose) pairs, the full variance matrix, and all simulated traces.

**`Learner`** (`learner.py`) — ABC selection and perturbation:

```python
from active_learning.learner import Learner
from active_learning.utils import generate_dynamic_targets

targets = generate_dynamic_targets(list(circuit_dict.values()))
learner = Learner(circuit_dict, targets, config)
belief_cloud, best_nmse = learner.update_belief(belief_cloud, promo_params, lab_data_dict)
```

**`gcad_utils`** (`gcad_utils.py`) — Bridge between the AL and GCAD:

```python
from active_learning.gcad_utils import run_gcad_miner, extract_top_circuits, update_gcad_library

run_gcad_miner("gcad_inputs/FreeSearch_settings.json", "results/cycle_0/")
circuits = extract_top_circuits("results/cycle_0/", m_target=3)
update_gcad_library(calibrated_parts)   # overwrites gcad/data/parts.pkl (backup created automatically)
```

---

## Notebooks

Run the notebooks in `notebooks/` in order for a full guided walkthrough. Each notebook produces output files consumed by later steps.

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_extract_circuits` | Parse GCAD Pareto front, select M distinct topologies | `selected_M_circuits.pkl` |
| `02_define_ground_truth` | Define synthetic "hidden truth" by mutating nominal parameters | `ground_truth/true_parts.pkl` |
| `03_generate_prior` | Sample a belief ensemble around nominal parameters | `al_memory/prior_belief_cloud_*.pkl` |
| `04_design_experiments` | Run OED grid, visualize information heatmap | `al_memory/designer_decisions_cycle_0.pkl` |
| `05_run_active_learning` | Run the full AL loop, plot convergence, save calibrated params | `al_memory/calibrated_parts.pkl` |
| `06_gcad_rerun` | Update GCAD library + rerun circuit mining with calibrated model | `results/Miner_PostAL_*/` |
| `07_run_api` | Minimal API demo (3 cells) | — |
| `08_full_loop` | Full autonomous GCAD → AL → GCAD loop | — |

**Working directory:** all notebooks in `notebooks/` expect to be run with the **repo root** as the working directory. In Jupyter, set this via `%cd /path/to/ActiveLearning-GCAD` in the first cell, or launch Jupyter from the repo root.

### `single_circuit_example/`

Single-circuit reference notebook. Show the AL internals (OED landscape, per-cycle belief cloud) without the `ActiveLearningLoop` wrapper. Useful for understanding each component or adapting the framework.

- `single_circuit_al.ipynb` — General AL demo on one circuit


---

## Quick Start

For the fastest path to results, open **`example_walkthrough.ipynb`** in the repo root. It runs the complete pipeline end-to-end in a single notebook with inline explanations.

For a full step-by-step walkthrough with intermediate outputs and visualizations, run `notebooks/01` through `notebooks/06` in order.

---

## How It Works

### 1. Prior Generation

A belief ensemble of N parameter dictionaries is sampled around the nominal (textbook) values using a log-normal distribution. Each model is a plausible hypothesis for the true biology.

### 2. Optimal Experimental Design

For each (circuit, dosage) pair, the Designer simulates all N models and computes the time-averaged prediction variance. High variance = high disagreement = informative experiment. The top-budget pairs are selected.

### 3. Virtual Lab

The Lab simulates the selected experiments using the *true* parameters plus Gaussian noise. In a real deployment, this step is replaced by actual wet-lab measurements.

### 4. ABC Inference

The Learner scores every model in the ensemble against all accumulated lab data using total NMSE. The top-20% survivors are resampled with exponentially-decaying perturbation (analogous to simulated annealing). The ensemble narrows toward the true parameters over cycles.

### 5. Library Update

`update_gcad_library()` overwrites `gcad/data/parts.pkl` with the posterior mean parameters. A backup of the nominal values is created automatically on first call. GCAD can then be re-run to discover circuits optimized for the real biology.

---

## Citation

If you use this framework and wish to cite, please use the following BibTex and cite this repository.

```bash
@article{dreyer-2026,
	author = {Dreyer, Kathleen S. and Nguyen, Anh V. and Bora, Gauri G. and Redus, Lauren E. and Edelstein, Hailey I. and Garcia, Jocelyn J. and Anastasia, Eleftheria and Dray, Kate E. and Leonard, Joshua N. and Mangan, Niall M.},
	journal = {ACS Synthetic Biology},
	month = {2},
	number = {3},
	pages = {1033--1052},
	title = {{GCAD: A Computational Framework for Mammalian Genetic Program Computer-Aided Design}},
	volume = {15},
	year = {2026},
	doi = {10.1021/acssynbio.5c00670},
	url = {https://doi.org/10.1021/acssynbio.5c00670},
}
```
---

### Questions? 

Please contact [pqkuang@gmail.com](mailto:pqkuang@gmail.com)
