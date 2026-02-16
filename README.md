# Fish Population Modelling — Spatial Reaction–Diffusion

Numerical modelling of single-species fish population dynamics, progressing from
time-only ODEs to a 1D spatial reaction–diffusion PDE with harvesting.

## Notebook

The primary notebook is **`visualsv3.ipynb`**. It is designed to run
top-to-bottom (`Restart & Run All`) with no external data files.

### Sections

| Section | Model | Equation |
|---------|-------|----------|
| **1** | Logistic growth (time-only ODE) | `dx/dt = r x (1 - x/K)` |
| **2** | Logistic growth + fishing | `dx/dt = r x (1 - x/K) - h x` |
| **3A** | Pure diffusion (spatial sanity check) | `u_t = D u_ss` |
| **3B** | Reaction–diffusion (growth + migration) | `u_t = r u (1 - u/K) + D u_ss` |

Each section validates against known solutions before adding complexity.

### Section Details

**1 — Logistic Fish Population Model**
- Compares RK45 numerical integration against the closed-form logistic solution
- Validation suite across 4 parameter sets; positivity checks
- Plots: multiple initial conditions converging to K; growth-rate sweep

**2 — Logistic Growth with Fishing**
- Proportional harvesting at rates `h = [0, 0.10, 0.30, 0.45, 0.55]`
- For `h < r`: equilibrium drops to `K(1 - h/r)`; numeric vs analytic validated
- For `h >= r`: population collapses (overfishing)

**3A — Pure Diffusion Sanity Check**
- 1D domain `s in [0, 600]` (offshore distance), 301 grid points
- Finite-difference Laplacian with Neumann (no-flux) boundary conditions
- Explicit Euler time-stepping with stability-guided dt
- Verifies mass conservation (no-flux + no reaction = constant total mass)

**3B — Reaction–Diffusion (No Fishing)**
- Combines logistic growth with spatial diffusion on the same grid as 3A
- Part A: non-uniform Gaussian IC — growth fills tails while diffusion smooths
- Part B: uniform IC validation — confirms PDE reduces to logistic ODE when
  spatially homogeneous (max spatial deviation ~ machine epsilon)

## Spatial Domain

The 1D coordinate `s` represents **offshore distance**:
- `s = 0` — coastline (shore boundary)
- `s = L = 600` — far offshore boundary

Boundary conditions are **no-flux (Neumann)**: `du/ds = 0` at both ends,
meaning no fish leave the domain. The Laplacian uses ghost-point equivalence
at the boundaries.

## Numerical Methods

| Method | Where used |
|--------|-----------|
| RK45 (adaptive Runge–Kutta) | Sections 1, 2 (ODEs via `scipy.integrate.solve_ivp`) |
| Explicit Euler + finite differences | Sections 3A, 3B (PDEs) |

The explicit Euler PDE solver uses the diffusion stability condition
`dt <= ds^2 / (2D)` with a safety factor of 0.45.

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- SciPy

## Generated Figures

The notebook saves the following PNG files (200 dpi):

| File | Content |
|------|---------|
| `step1_logistic_quick_view.png` | Quick-look logistic growth curve |
| `step1_logistic_x0_comparison.png` | Multiple initial conditions (numeric vs analytic) |
| `step1_logistic_r_sweep.png` | Growth-rate sweep comparison |

Additional plots are displayed inline but not saved to disk.

## Project Files

```
visualsv3.ipynb          Main notebook (current)
visuals.ipynb            Earlier version (v1)
visualsv2.ipynb          Earlier version (v2)
README.md                This file
```
