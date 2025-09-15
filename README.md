# 3drvm

This repository contains the **JAX implementation** of the **Implicit Deep Random Vortex Network (3D-iDRVN)**, a neural network–based Random Vortex Method for simulating incompressible fluid flows in **three-dimensional, wall-bounded domains**.

Unlike classical numerical solvers (finite differences, finite elements, spectral methods), this approach is **grid-free** and avoids explicit evaluation of the **Biot–Savart kernel**, making it suitable for geometrically complex domains, although, for now, we implemented the method just for the three-dimensional torus.

The method combines a **probabilistic vortex representation of the Navier–Stokes equations** with deep neural networks and a novel **loss function**.

---

## ✨ Features

- **Grid-free** simulation of incompressible 3D flows
- **Neural-network vorticity approximation** with implicit velocity recovery
- **Custom loss function** derived from vortex representation formula
- **Monte Carlo training scheme** for efficiency and flexibility
- Fully implemented in [JAX](https://github.com/jax-ml/jax) with GPU/TPU acceleration

---

## 👩‍💻 Authors

The authors of this project are:

- **Giuseppe Bruno** [University of Bern](https://www.imsv.unibe.ch/about_us/staff/bruno_giuseppe/index_eng.html)
- **Paula Cordero-Encinar** [Imperial College London](https://statml.io/students/paula-cordero-encinar/)
- **Filippo Giovagnini** [Imperial College London](https://profiles.imperial.ac.uk/f.giovagnini23)

Please note that the commit history of this repository does not accurately reflect the relative contributions of the authors, as the codebase originates from the migration of a prior project.

---

## 📚 Background

This project generalizes the **Deep Random Vortex Method (DRVM)** [Qi & Meng, 2022] and its implicit variant [Cherepanov, 2024] to **three-dimensional flows**.

### References
- Sirignano, J., Spiliopoulos, K. *DGM: A deep learning algorithm for solving PDEs.* J. Comput. Phys., 2018.  
- Qi, J., Meng, X. *Deep Random Vortex Method for 2D Navier–Stokes Equations.* 2022.  
- Cherepanov, A. *Neural networks based random vortex methods.* 2024.  
- Cherepanov, A. *A Monte Carlo method for incompressible fluid dynamics.* 2023.  
- Giovagnini, F., Crisan, D. *A uniform particle approximation to the Navier–Stokes-alpha models in three dimensions with advection noise.* 2025.

---

## ⚙️ Installation

No need to intall any dependecies, thanks to the [astral-sh/uv](https://github.com/astral-sh/uv) Python package and project manager! Installation is as simple as this. First of all do:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or
```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then you can just do:
```bash
git clone https://github.com/filippogiovagnini/3drvm.git
cd 3drvm/src
uv run experiment.py
```
without worrying about dependencies between libraries.

## 📂 Repository Structure

```bash
3drvm/
│
├── src/                  # Core implementation in JAX
│   ├── train.py          # Neural network
│   ├── update_stepper.py           # Loss function
│   ├── interpolators.py     # Loss function
│   ├── num_solvers.py         # Vortex/SDE solver
│   └── utilities.py          # Helpers
├── experiment.py                  # Core implementation in JAX
└── README.md
```