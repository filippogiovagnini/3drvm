# 3drvm

This repository contains the **JAX implementation** of the **Implicit Deep Random Vortex Network (3D-iDRVN)**, a neural network–based Random Vortex Method for simulating incompressible fluid flows in **three-dimensional, wall-bounded domains**.

Unlike classical numerical solvers (finite differences, finite elements, spectral methods), this approach is **grid-free** and avoids explicit evaluation of the **Biot–Savart kernel**, making it suitable for geometrically complex domains.  

The method combines a **probabilistic vortex representation of the Navier–Stokes equations** with deep neural networks and a novel **loss function**.

---

## ✨ Features

- **Grid-free** simulation of incompressible 3D flows  
- **Neural-network vorticity approximation** with implicit velocity recovery  
- **Custom loss function** derived from vortex representation formula  
- **Monte Carlo training scheme** for efficiency and flexibility  
- Fully implemented in **JAX** with GPU/TPU acceleration  

---

## 📚 Background

This project generalizes the **Deep Random Vortex Method (DRVM)** [Qi & Meng, 2022] and its implicit variant [Cherepanov, 2024] to **three-dimensional flows**.

### References
- Sirignano, J., Spiliopoulos, K. *DGM: A deep learning algorithm for solving PDEs.* J. Comput. Phys., 2018.  
- Qi, J., Meng, X. *Deep Random Vortex Method for 2D Navier–Stokes Equations.* 2022.  
- Cherepanov, A. *Neural networks based random vortex methods.* 2024.  
- Cherepanov, A. *A Monte Carlo method for incompressible fluid dynamics.* 2023.  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
pip install uv
git clone https://github.com/your-username/3drvm.git
cd 3drvm/src
uv run experiment1.py
