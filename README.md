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
| **3C** | Reaction–diffusion + fishing policy | `u_t = r u (1 - u/K) - h(s,t) u + D u_ss` |

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

**3C — Reaction–Diffusion with Fishing Policy (200-Mile Boundary)**
- Introduces spatially varying harvesting `h(s,t)`: fishing intensity differs
  inside (`s <= 200`) vs outside (`s > 200`) a 200-mile economic zone
- Four policy scenarios are compared:

| Scenario | h_in | h_out | Description |
|----------|------|-------|-------------|
| **A** | 0.0 | 0.0 | No fishing (baseline, identical to 3B) |
| **B** | 0.2 | 0.2 | Uniform fishing everywhere |
| **C** | 0.0 | 0.2 | 200-mile ban: inshore protected, offshore fished |
| **D** | 0.0 | 0.2 + pulse | Same as C, but with an intense offshore pulse (`h=0.8` for `t in [20,25]`) |

- **Key results:**
  - **A:** Growth fills domain to K (no harvesting pressure)
  - **B:** Uniform fishing depresses equilibrium below K everywhere
  - **C:** Inshore reaches K, offshore is depressed; diffusion creates a
    **spillover gradient** from protected to fished zones
  - **D:** Intense pulse causes a transient biomass crash, followed by partial
    recovery once the pulse ends
- Tracks inshore biomass, offshore biomass, total biomass, instantaneous catch
  rate, and cumulative catch for each scenario

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
| Explicit Euler + finite differences | Sections 3A, 3B, 3C (PDEs) |

The explicit Euler PDE solver uses the diffusion stability condition
`dt <= ds^2 / (2D)` with a safety factor of 0.45. Section 3C adds an
additional dt constraint based on the maximum reaction rate to prevent
instability from the harvesting terms.

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
| `step3c_snapshots_A.png` | Density snapshots — Scenario A (no fishing) |
| `step3c_snapshots_B.png` | Density snapshots — Scenario B (uniform fishing) |
| `step3c_snapshots_C.png` | Density snapshots — Scenario C (200-mile ban) |
| `step3c_snapshots_D.png` | Density snapshots — Scenario D (ban + pulse) |
| `step3c_biomass_A.png` | Biomass time series — Scenario A |
| `step3c_biomass_B.png` | Biomass time series — Scenario B |
| `step3c_biomass_C.png` | Biomass time series — Scenario C |
| `step3c_biomass_D.png` | Biomass time series — Scenario D |
| `step3c_catch_A.png` | Catch rate & cumulative catch — Scenario A |
| `step3c_catch_B.png` | Catch rate & cumulative catch — Scenario B |
| `step3c_catch_C.png` | Catch rate & cumulative catch — Scenario C |
| `step3c_catch_D.png` | Catch rate & cumulative catch — Scenario D |

## Project Files

```
visualsv3.ipynb          Main notebook
README.md                This file
```
