# PI-A-KMNet: Physics-Informed Alternating KAN–MLP Networks for Ill-Posed Nonlinear Wave Equations

Code for the paper *"A physics-informed hybrid architecture of alternating KAN and MLP for ill-posed nonlinear equations with linearly unstable waves"*.

This repository implements two physics-informed hybrid architectures, **PI-A-KMNet**<sub>MLP</sub> and **PI-A-KMNet**<sub>KAN</sub>, which alternate MLP (multilayer perceptron) layers and KAN (Kolmogorov–Arnold network, degree-4 Chebyshev) layers, and compares them with two pure baselines on two linearly unstable nonlinear wave equations: the **"bad" Boussinesq equation** and the **"bad" Jaulent–Miodek (JM) equation**.

## Network naming convention

All experiment folders use the following names for the four compared methods:

| Folder name | Paper notation | Description |
|---|---|---|
| `DNN` | PINN | pure-MLP physics-informed network |
| `KAN` | PI-KAN | pure-KAN physics-informed network |
| `DNN-KAN` | PI-A-KMNet<sub>MLP</sub> | alternating network starting with an MLP layer |
| `KAN-DNN` | PI-A-KMNet<sub>KAN</sub> | alternating network starting with a KAN layer |

## Repository structure

### `Bad-Boussinesq-equation/`

- **`solitary-wave/`** — forward problem with an analytical solution:
  - `3rd-soliton/` — third-order solitary wave solution (collocation points in `*-residualpoints/`, results in `*-results/` under the 5,000 / 10,000 / 30,000 trainable-parameter regimes, degree-4 KAN).
- **`gauss wave packet/`** — forward problems **without** analytical solutions:
  - `BQ-gauss/` — single Gaussian wave packet; the four methods plus `input data/` (fixed LHS collocation points) and `spectral method/` (band-limited regularized Fourier pseudo-spectral reference solution, following Charlier et al., *Appl. Numer. Math.* **217** (2025) 216–233, with RK4 time stepping);
  - `BQ-2gauss/` — double Gaussian wave packet; same layout.

### `Bad-JM-equation/`

- **`solitary-wave/`** — forward problems with analytical solutions: `jmcase1/` and `jmcase2/` (analytical solution data plus results of the four methods).
- **`inverse problem/`** — five-parameter inversion for the `bad` JM equation:
  - `jm-inverse-case2/` — noise-free observational data;
  - `jm-inverse-case2-noise5pct/` — observational data corrupted with i.i.d. Gaussian noise of 5% of the standard deviation of each field.
- **`gauss wave packet/`** — Gaussian wave packet without analytical solution: the four methods plus `ADM/` (independent reference solution computed with the piecewise Adomian decomposition method, band-limited with a super-Gaussian cutoff).

## Getting started

### Prerequisites

- Python 3.x
- PyTorch, NumPy, SciPy, Matplotlib

### Usage

Each experiment is a self-contained Jupyter notebook inside its result folder; open the notebook of interest and run it top to bottom. Reference-solution scripts/notebooks are kept in the corresponding `spectral method/` and `ADM/` folders.

## Data

Additional result data are available at: <https://drive.google.com/file/d/1Zzzt3FyIkLd399oatPi4OJD5ixEMC6Hj/view?usp=drive_link>

## Citation

If you use this code, please cite the paper above.

## License

For usage permissions, please contact the repository maintainer.

---

Last updated: 2026-10-04
