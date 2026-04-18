# RNN Honours Rotation — System Identification of a Nonlinear Two-Cart Spring System

## Overview

This project investigates **system identification** of a nonlinear two-mass spring-damper system, comparing a classical **Least-Squares (LS)** baseline against **Recurrent Neural Network (RNN)** approaches. The work is inspired by Revay, Wang & Manchester (IEEE Control Systems Letters, 2021) but focuses on the standard (non-convex) formulation.

The central challenge: the coupling spring between the two masses has a **piecewise-linear hardening nonlinearity** (a soft zone near equilibrium, stiffer outside), which breaks the assumptions of linear system identification. The goal is to learn a model that can reproduce the system's trajectories autonomously (closed-loop / autoregressive rollout) from input-output data alone.

---

## Physical System

Two masses connected by springs and dampers, driven by an external force on mass 1:

| Parameter | Value |
|-----------|-------|
| Mass 1 (m₁) | 1/4 kg |
| Mass 2 (m₂) | 1/3 kg |
| Damping 1 (c₁) | 1/4 Ns/m |
| Damping 2 (c₂) | 1/3 Ns/m |
| Spring 1 (k₁) | 1.0 N/m (linear) |
| Spring 2 (k₂) | 5/6 N/m (nonlinear hardening) |

**Hardening spring nonlinearity** (soft zone ±δ):

```
γ(d) = k₂ · d             if |d| ≤ δ
γ(d) = k₂ · (2d − δ)      if d > δ
γ(d) = k₂ · (2d + δ)      if d < −δ
```

Two variants were studied: original soft zone δ = 1.0 m, and a shrunk soft zone δ = 0.25 m (making the nonlinearity more prominent).

**Input signal**: Piecewise-constant force on mass 1, switching every 5 s with Gaussian amplitudes (σ = 2–3 N). Simulation: RK4, Δt = 0.01 s, T = 200 s → 20,000 time steps.

**Coordinate transform**: z = [x₁ − x₂, v₁ − v₂, x₂, v₂]ᵀ (relative coordinates), which diagonalises the linear part and isolates the nonlinearity in z₁.

---

## Repository Structure

```
RNN Honours Rotation/
│
├── README.md                        ← This file
├── Final_Report.tex                 ← LaTeX source for the written report
├── Final_Report.pdf                 ← Compiled PDF report
│
├── src/                             ← Standalone Python scripts
│   ├── simulate_2_springs_simple.py     Generate nonlinear simulation data (seed=42)
│   ├── simulate_2_springs_linear.py     Generate linear spring data for LS validation
│   ├── system_id.py                     LS identification on raw (x) states
│   ├── system_id_transformed.py         LS identification on z-coordinates
│   ├── system_id_linear_springs.py      LS on linear data (sanity check)
│   ├── Vanilla RNN.py                   Open-loop RNN baseline (input u → outputs)
│   └── Can be discarded/            ← Deprecated / superseded scripts
│       ├── simulate_2_springs.py        Earlier simulator (replaced by simple version)
│       └── Improved RNN (old).py        Earlier RNN attempt (superseded by notebooks)
│
├── notebooks/                       ← Jupyter notebooks (main experimental work)
│   ├── Phase1_Data_and_LS_Baselines.ipynb   Phase 1: train/test splits, shrunk springs, LS NSE table
│   ├── Phase2_RNN_Experiments.ipynb         Phase 2: LR scheduler, Exp 2a (CL-RNN) & 2b (residual)
│   ├── CL_RNN_Transformed_data.ipynb        CL-RNN with ReLU activation, z-coords
│   ├── CL_ELU_RNN_Transformed_data.ipynb    CL-RNN with ELU activation
│   ├── CL_tanh_RNN_Transformed_data.ipynb   CL-RNN with tanh activation
│   ├── CL_NLT_RNN_Transformed_data.ipynb    CL-RNN with deadzone (NLT) pre-processing
│   ├── Data_Transform.ipynb                 Simulate + save z-coordinate transformed data
│   ├── Open Loop Vanilla RNN.ipynb          Open-loop RNN baseline exploration
│   ├── Closed Loop Vanilla RNN.ipynb        Closed-loop RNN on nonlinear system
│   └── Linear Closed Loop Vanilla RNN.ipynb Closed-loop RNN on linear system (sanity check)
│
├── data/                            ← Saved simulation and LS matrices (.npz)
│   ├── simulation_data.npz                  Train trajectory (seed=42): y_vals [20000×4], u_plot
│   ├── transformed_simulation_data.npz      Train trajectory in z-coords: z_vals, u_plot
│   ├── linear_simulation_data.npz           Linear spring train trajectory: y_vals, u_plot
│   ├── test_simulation_data.npz             Test trajectory (seed=99): y_vals, u_plot
│   ├── test_transformed_simulation_data.npz Test trajectory in z-coords
│   ├── shrunk_train_simulation_data.npz     Shrunk δ=0.25 train trajectory: y_vals, u_plot
│   ├── shrunk_train_transformed_data.npz    Shrunk train trajectory in z-coords
│   ├── shrunk_test_simulation_data.npz      Shrunk δ=0.25 test trajectory: y_vals, u_plot
│   ├── shrunk_test_transformed_data.npz     Shrunk test trajectory in z-coords
│   ├── ls_matrices_original_raw.npz         LS matrices: original δ=1.0, raw x states
│   ├── ls_matrices_original_transformed.npz LS matrices: original δ=1.0, z states
│   ├── ls_matrices_shrunk_raw.npz           LS matrices: shrunk δ=0.25, raw x states
│   └── ls_matrices_shrunk_transformed.npz   LS matrices: shrunk δ=0.25, z states
│
├── figures/                         ← Generated plots
│   ├── simulate_2_springs_simple.png
│   ├── simulation_2_springs_linear.png
│   ├── linear_model_comparison.png
│   ├── linear_springs_model_comparison.png
│   ├── transformed_linear_model_comparison.png
│   ├── Vanilla RNN Comparison.png
│   └── Used in the report/          ← Curated figures included in Final_Report.pdf
│       ├── Original non linear springs with delta 1.png
│       ├── Shrunk non linear springs.png
│       ├── 2 spring regions.png
│       ├── LS on original non linear springs raw states.png
│       ├── LS on original non linear springs transformed z states.png
│       ├── LS on shrunk non linear springs.png
│       ├── LS on shrunk non linear springs with transformed.png
│       ├── CL RNN Loss.png
│       ├── CL RNN Trajectory.png
│       ├── CL ELU RNN Loss.png
│       ├── CL ELU RNN Trajectory.png
│       ├── CL tanh RNN Trajectory.png
│       ├── CL NLT RNN Trajectory.png
│       ├── 2a train loss.png
│       ├── 2a eval.png
│       ├── 2b Loss.png
│       └── 2b eval.png
│
└── docs/                            ← Reference material
    ├── A_Convex_Parameterization_of_Robust_Recurrent_Neural_Networks.pdf
    └── pdf_content.txt              ← Extracted text from the reference paper
```

---

## How to Run

### Prerequisites

```bash
pip install numpy matplotlib scipy torch jupyter
```

### 1. Generate simulation data

```bash
cd src/
python simulate_2_springs_simple.py    # → simulation_data.npz
python simulate_2_springs_linear.py   # → linear_simulation_data.npz
```

### 2. Run classical LS system identification

```bash
python system_id.py                   # LS on raw x states
python system_id_transformed.py       # LS on z-coordinates
python system_id_linear_springs.py    # LS sanity check on linear system
```

### 3. Run notebooks (recommended order)

1. `Data_Transform.ipynb` — generate z-coordinate transformed data
2. `Phase1_Data_and_LS_Baselines.ipynb` — full LS baseline study with train/test splits
3. `Open Loop Vanilla RNN.ipynb` — open-loop RNN baseline
4. `Closed Loop Vanilla RNN.ipynb` — closed-loop RNN on nonlinear system
5. `Linear Closed Loop Vanilla RNN.ipynb` — closed-loop RNN sanity check on linear system
6. `CL_RNN_Transformed_data.ipynb` — closed-loop ReLU RNN on z-coords
7. `CL_ELU_RNN_Transformed_data.ipynb` — ELU variant
8. `CL_tanh_RNN_Transformed_data.ipynb` — tanh variant
9. `CL_NLT_RNN_Transformed_data.ipynb` — deadzone (NLT) variant
10. `Phase2_RNN_Experiments.ipynb` — full Phase 2 with LR scheduler and physics-informed residual

---

## Methods

### Classical LS (ARMA model)

The one-step-ahead linear model in z-coordinates:

```
z_{k+1} = D z_k + E₁ u_{k+1} + E₂ u_k
```

Solved in closed form via pseudoinverse: Θ = Z_next · Z_reg^T · (Z_reg · Z_reg^T)⁻¹

This works well for the linear system and the shrunk-spring variant (smaller nonlinearity), but diverges under autonomous rollout for the original δ = 1.0 system.

### Closed-Loop RNN (Elman)

An Elman RNN trained to predict z_{k+1} from [z_k; u_k], with its own prediction fed back as input (no teacher forcing at evaluation):

```
h_{k+1} = σ(W_h h_k + W_x [z_k; u_k] + b_h)
ẑ_{k+1} = W_y h_{k+1} + b_y
```

Architecture: input size 5, hidden size 64. Trained with MSE loss, Adam optimiser.

### NLT (Deadzone) Pre-processing

Physics-inspired pre-processing: before the hidden update, z₁ (relative position) is passed through a deadzone function with threshold δ = 1.0, zeroing out the soft spring region:

```
deadzone(z₁, δ) = 0           if |z₁| ≤ δ
                = z₁ − δ·sign(z₁)  otherwise
```

### Physics-Informed Residual RNN (Experiment 2b)

The RNN input is augmented with the LS one-step prediction ẑ^LS_{k+1}, giving a 9-dimensional input [z_k; u_k; ẑ^LS_{k+1}]. The network learns only the nonlinear residual on top of the LS model. This was the best-performing configuration.

---

## Results Summary

### Phase 1 — LS Baselines (NSE = ‖ŷ − y‖ / ‖y‖)

| Variant | States | Train NSE | Test NSE |
|---------|--------|-----------|----------|
| Original δ=1.0 | raw x | 0.1693 | 0.2001 |
| Original δ=1.0 | z-coords | 0.2553 | 0.3089 |
| Shrunk δ=0.25 | raw x | 0.0631 | 0.0795 |
| Shrunk δ=0.25 | z-coords | 0.1181 | 0.1542 |

NSE < 1 means the model beats the zero predictor. The LS model with shrunk springs achieves good accuracy, confirming the nonlinearity is the dominant challenge.

### Phase 2 — RNN Experiments

| Experiment | Architecture | Train NSE (z1/z3) | Test NSE (z1/z3) |
|------------|-------------|-------------------|-----------------|
| 2a: CL-RNN + LR scheduler | ReLU, h=64, ReduceLROnPlateau | 1.748 / 1.766 | 1.745 / 1.552 |
| 2b: Physics-informed residual | ReLU, h=64, 9-dim input | **0.348 / 0.185** | **0.399 / 0.201** |

Experiment 2a diverges under autonomous rollout (NSE > 1), illustrating the train/inference gap. Experiment 2b, which uses the LS prediction as an additional input, generalises well and achieves NSE < 0.4 on both train and test sets.

---

## Key Findings

- Linear LS fails to capture the hardening spring nonlinearity under autonomous rollout, but performs surprisingly well one-step-ahead (low NSE) for the shrunk spring variant.
- Vanilla closed-loop RNN training is unstable: low teacher-forced training loss does not guarantee autonomous stability.
- **Physics-informed residual learning** (Exp. 2b) is the most effective approach: providing the LS prediction as a feature dramatically stabilises autonomous rollout and reduces NSE by ~5× compared to standalone CL-RNN.
- The ReduceLROnPlateau scheduler anneals the learning rate from 0.01 to ~3.9×10⁻⁵, showing the importance of learning rate management for long training runs.

---

## Report

The written report (`Final_Report.pdf`) follows the McGill Honours Rotation format: 1-page body (0.5 in margins, 12pt serif, 14pt title), references on page 2, and figures/tables in the appendix. Compile from source:

```bash
pdflatex Final_Report.tex
```

---

## References

- Revay, M., Wang, R., & Manchester, I. R. (2021). Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness. *IEEE Control Systems Letters*, 5(6), 2096–2101.
- Project code: Obed Gundra, McGill University Honours Rotation, 2025–2026.
