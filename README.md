# Fish Population Modelling — Spatial Reaction-Diffusion with Harvesting

Numerical modelling of single-species fish population dynamics, progressing from
time-only ODEs to a 1-D spatial reaction-diffusion PDE with spatially varying
harvesting. The project builds complexity incrementally: each section validates
against known solutions or limiting cases before the next layer is added.

All equations are presented and all results are displayed in **dimensionless
form** throughout (see [Non-Dimensionalisation](#non-dimensionalisation) below).

## Notebook

The primary notebook is **`visualsv3.ipynb`**. It imports model, solver,
validation, and plotting functions from the companion Python modules
(`ode_models.py`, `pde_solver.py`, `validation.py`, `plotting.py`).
Run top-to-bottom (`Restart & Run All`) with no external data files.

---

## Non-Dimensionalisation

All sections use a common dimensionless scaling:

| Substitution | Definition | Range |
|---|---|---|
| $\tilde{x} = x/K$ (ODE) or $x = u/K$ (PDE) | Density scaled by carrying capacity | $[0, 1]$ at equilibrium |
| $\xi = s/L$ | Offshore position scaled by domain length | $[0, 1]$ |
| $\tau = rt$ | Time scaled by growth timescale $1/r$ | $[0, \infty)$ |
| $\tilde{h} = h/r$ | Harvest rate scaled by growth rate | $[0, 1]$ for sustainable fishing |
| $\delta = D/(rL^2)$ | Dimensionless diffusion parameter | — |

The parameter $\delta$ compares the diffusion timescale $L^2/D$ to the growth timescale $1/r$:
- **Small $\delta \ll 1$:** growth is fast relative to mixing — spatial gradients persist.
- **Large $\delta \gg 1$:** fish mix rapidly — solution tends to spatial homogeneity.

With the project's base parameters ($D=10$, $r=0.5$, $L=600$):
$$\delta = \frac{D}{rL^2} = \frac{10}{0.5 \times 360000} \approx 5.56 \times 10^{-5}$$

---

## 1 — Logistic Fish Population Model (Time-Only ODE)

### Governing Equation (Dimensionless)

Setting $\tilde{x} = x/K$ and $\tau = rt$, the logistic ODE reduces to:

$$
\frac{d\tilde{x}}{d\tau} = \tilde{x}(1 - \tilde{x})
$$

This ODE has **no free parameters** — $r$ and $K$ only enter via the initial
condition $\tilde{x}_0 = x_0/K$ and the timescale $\tau = rt$.

### Analytic Solution

$$
\tilde{x}(\tau) = \frac{1}{1 + \left(\dfrac{1 - \tilde{x}_0}{\tilde{x}_0}\right) e^{-\tau}}
$$

Equilibria: $\tilde{x}^* = 0$ (unstable) and $\tilde{x}^* = 1$ (globally attracting for $\tilde{x}_0 > 0$).

### Numerical Method

RK45 adaptive Runge-Kutta via `scipy.integrate.solve_ivp` with `rtol=1e-9`,
`atol=1e-12`. Implemented by passing `r=1`, `K=1`, `x0=x̃₀` to the existing
`logistic_analytic` / `logistic_numeric` helpers — these then directly produce
the ND solution $\tilde{x}(\tau)$.

### Validation

Four dimensional test cases $(r, K, x_0)$ are converted to their ND equivalents
$\tilde{x}_0 = x_0/K$. All run with `r=1, K=1` over a common $\tau$-grid up to
$\tau = 15$. Maximum absolute errors and positivity are reported.

| Original case | $r$ | $K$ | $x_0$ | $\tilde{x}_0 = x_0/K$ |
|---------------|-----|-----|--------|------------------------|
| 1 | 0.3 | 500 | 50 | 0.10 |
| 2 | 0.8 | 1000 | 100 | 0.10 |
| 3 | 1.2 | 1200 | 1500 | 1.25 |
| 4 | 0.5 | 300 | 290 | 0.967 |

**Key insight:** Cases 1 and 2 have identical ND dynamics ($\tilde{x}_0 = 0.10$),
demonstrating that $r$ and $K$ individually are irrelevant once scaled — only
$\tilde{x}_0$ matters.

### Plots

| Figure | Description |
|--------|-------------|
| Quick-look logistic curve | $\tilde{x}(\tau)$ for $\tilde{x}_0 = 0.1$, $\tau \in [0, 18]$ |
| Initial-condition comparison | RK45 vs analytic for $\tilde{x}_0 \in \{0.1, 0.5, 1.25\}$ |

---

## 2 — Logistic Growth with Fishing (Harvesting)

### Governing Equation (Dimensionless)

With dimensionless harvest rate $\tilde{h} = h/r$:

$$
\frac{d\tilde{x}}{d\tau} = \tilde{x}(1 - \tilde{x}) - \tilde{h}\,\tilde{x} = \tilde{x}\!\left(1 - \tilde{x} - \tilde{h}\right)
$$

### Analytic Behaviour

| Regime | Condition | ND equilibrium |
|--------|-----------|----------------|
| Sustainable | $\tilde{h} < 1$ | $\tilde{x}^* = 1 - \tilde{h}$ |
| Critical | $\tilde{h} = 1$ | $\tilde{x}^* = 0$ (marginal collapse) |
| Overfishing | $\tilde{h} > 1$ | $\tilde{x} \to 0$ (extinction) |

**Maximum Sustainable Yield:** $\tilde{h}_{\mathrm{MSY}} = 1/2$, achieved at $\tilde{x}^*_{\mathrm{MSY}} = 1/2$.

### Parameters

| Symbol | Dimensional value | Dimensionless equivalent |
|--------|------------------|--------------------------|
| $r$ | 0.50 | — (absorbed into $\tau$) |
| $K$ | 1000 | — (absorbed into $\tilde{x}$) |
| $x_0$ | 500 | $\tilde{x}_0 = 0.5$ |
| $h$ values tested | 0.00, 0.10, 0.30, 0.45, 0.55 | $\tilde{h} = $ 0.00, 0.20, 0.60, 0.90, 1.10 |
| Time span | $[0, 40]$ dimensional | $\tau \in [0, 20]$ |

### Key Results

| $\tilde{h}$ | Equilibrium $\tilde{x}^* = 1 - \tilde{h}$ | Outcome |
|------------|---------------------------------------------|---------|
| 0.00 | 1.00 | No fishing — converges to carrying capacity |
| 0.20 | 0.80 | Sustainable harvest at 80% of capacity |
| 0.60 | 0.40 | Moderate depletion |
| 0.90 | 0.10 | Heavy depletion — near collapse |
| 1.10 | — | Overfishing — extinction |

Dashed horizontal lines mark each analytic equilibrium $\tilde{x}^* = 1 - \tilde{h}$.

---

## 3A — Pure Diffusion Sanity Check (Migration Only)

### Governing Equation (Dimensionless)

Using $x = u/K$, $\xi = s/L$, $\tau = r_{\rm ref}\,t$ (reference $r_{\rm ref} = 0.5$):

$$
\frac{\partial x}{\partial \tau} = \delta\,\frac{\partial^2 x}{\partial \xi^2}, \qquad \xi \in [0,1]
$$

with no-flux boundaries:

$$
\frac{\partial x}{\partial \xi}(0,\tau) = 0, \qquad \frac{\partial x}{\partial \xi}(1,\tau) = 0
$$

### Purpose

Validate the spatial finite-difference discretisation and boundary treatment
**in isolation** before adding reaction or fishing terms. Expected behaviour:

1. An initial Gaussian bump (centred at $\xi = 0.25$) spreads and flattens.
2. Total dimensionless mass $M(\tau) = \int_0^1 x\,d\xi$ is conserved (no flux out, no reaction).

### Grid and ND Parameters

| Parameter | Value |
|-----------|-------|
| $\delta = D/(r_{\rm ref} L^2)$ | $\approx 5.56 \times 10^{-5}$ |
| $N$ (grid points) | 301 |
| $\Delta\xi$ | $1/(N-1) \approx 3.33 \times 10^{-3}$ |
| $\tau_{\rm end} = r_{\rm ref} \cdot 20$ | 10.0 |
| IC | $x(\xi, 0) = \exp\!\bigl(-\bigl((\xi - 0.25)/0.0833\bigr)^2\bigr)$ |

### Finite-Difference Laplacian

Interior points (standard second-order central difference):
$$
\frac{\partial^2 x}{\partial \xi^2}\bigg|_i \approx \frac{x_{i+1} - 2x_i + x_{i-1}}{\Delta\xi^2}
$$

Boundaries (ghost-point equivalence for Neumann conditions):
$$
\frac{\partial^2 x}{\partial \xi^2}\bigg|_0 \approx \frac{2(x_1 - x_0)}{\Delta\xi^2}, \qquad
\frac{\partial^2 x}{\partial \xi^2}\bigg|_{N-1} \approx \frac{2(x_{N-2} - x_{N-1})}{\Delta\xi^2}
$$

### Time Stepping

Explicit Euler with the diffusion-stability condition in ND variables:

$$
\Delta\tau \le \frac{\Delta\xi^2}{2\delta}
$$

A safety factor of 0.45 is applied (equivalent to the dimensional condition $\Delta t \le \Delta s^2/(2D)$).

### Validation

- **Density snapshots** at $\tau = 0, 2.5, 5, 10$ show the Gaussian spreading over $\xi \in [0,1]$.
- **Mass conservation** confirms $M(\tau) = \int_0^1 x\,d\xi$ remains constant to machine precision.

---

## 3B — Reaction-Diffusion: Logistic Growth + Migration (No Fishing)

### Governing Equation (Dimensionless)

$$
\frac{\partial x}{\partial \tau} = x(1-x) + \delta\,\frac{\partial^2 x}{\partial \xi^2}, \qquad
\frac{\partial x}{\partial \xi}(0,\tau) = \frac{\partial x}{\partial \xi}(1,\tau) = 0
$$

Same grid, boundary conditions, and numerical method as 3A.

### ND Parameters

| Parameter | Dimensional | Dimensionless |
|-----------|-------------|---------------|
| $r$ | 0.5 | — |
| $K$ | 1.0 | — |
| $D$ | 10.0 | $\delta \approx 5.56 \times 10^{-5}$ |
| $T$ | 60.0 | $\tau_{\rm end} = 30.0$ |

### Sub-experiments

#### Part A — Non-uniform Initial Condition (Gaussian Bump)

$$
x(\xi, 0) = \exp\!\bigl(-\bigl((\xi - 0.25)/0.0833\bigr)^2\bigr)
$$

**Behaviour:** logistic growth fills the low-density tails toward $x^* = 1$
while diffusion (small $\delta$) slowly smooths spatial gradients. Snapshots
at $\tau = 0, 5, 10, 20, 30$.

#### Part B — Uniform Initial Condition Validation

$x(\xi, 0) = 0.2$ everywhere.

**Purpose:** A spatially uniform IC gives $\partial^2 x/\partial\xi^2 = 0$ for
all time, so the PDE reduces to the ND logistic ODE $d\tilde{x}/d\tau = \tilde{x}(1-\tilde{x})$.
The PDE spatial mean is overlaid against `logistic_analytic(τ, r=1, K=1, x0=0.2)`.
Maximum spatial deviation remains at machine epsilon.

---

## 3C — Reaction-Diffusion with Fishing Policy (200-Mile EEZ Boundary)

### Governing Equation (Dimensionless)

$$
\frac{\partial x}{\partial \tau} = x(1-x) - \tilde{h}(\xi,\tau)\,x + \delta\,\frac{\partial^2 x}{\partial \xi^2}
$$

The dimensionless fishing rate is piecewise constant across the EEZ boundary
at $\xi_{\rm bnd} = 200/600 = 1/3$:

$$
\tilde{h}(\xi,\tau) = \begin{cases}
\tilde{h}_{\rm in}  & \xi \le 1/3 \quad \text{(EEZ — regulated)} \\[4pt]
\tilde{h}_{\rm out} & \xi > 1/3 \quad \text{(international waters — unregulated)}
\end{cases}
$$

### Harvest Rate Calibration (Dimensionless)

$$
\tilde{h}_{\rm MSY} = \tfrac{1}{2}, \qquad
\tilde{h}_{\rm MEY} = \tfrac{1}{2}\!\left(1 - \sqrt{1-f}\right), \quad f \sim \mathrm{Uniform}(0.80, 0.90)
$$

**ND equilibria:**
- No fishing: $x^* = 1$
- At MSY: $x^* = 1/2$
- At MEY: $x^* = 1 - \tilde{h}_{\rm MEY}$

### Parameters

| Parameter | Dimensional value | ND equivalent |
|-----------|------------------|---------------|
| $L, N, D, r, K$ | Same as 3B | $\delta \approx 5.56 \times 10^{-5}$ |
| $\xi_{\rm bnd}$ | $s_{\rm bnd} = 200$ | $1/3$ |
| $T$ | 60.0 | $\tau_{\rm end} = 30.0$ |

### Additional Stability Constraint

$$
\Delta\tau \le \frac{0.2}{\max(1,\,\tilde{h}_{\rm in,\,max},\,\tilde{h}_{\rm out,\,max})}
$$

(equivalent to the dimensional condition; the actual $\Delta\tau$ is the minimum with the diffusion stability limit).

### Three Policy Scenarios

| Scenario | $\tilde{h}_{\rm in}$ (EEZ) | $\tilde{h}_{\rm out}$ (international) | Description |
|----------|--------------------------|--------------------------------------|-------------|
| **A** | 0 | 0 | No fishing (baseline, identical to 3B) |
| **B** | $\tilde{h}_{\rm MEY}$ | $\tilde{h}_{\rm MEY}$ | Uniform MEY fishing everywhere |
| **C** | $\tilde{h}_{\rm MEY}$ | $\tilde{h}_{\rm MSY} = 1/2$ | EEZ managed at MEY, international at MSY |

### Key Results

**Scenario A (No Fishing):**
$x \to 1$ everywhere — identical to 3B.

**Scenario B (Uniform MEY):**
Equilibrium depressed to $x^* = 1 - \tilde{h}_{\rm MEY}$ uniformly,
sustaining near-optimal long-term yield below the MSY threshold.

**Scenario C (EEZ at MEY, International at MSY):**
EEZ ($\xi \le 1/3$) converges to $x^*_{\rm MEY} = 1 - \tilde{h}_{\rm MEY}$.
International waters ($\xi > 1/3$) converge to $x^*_{\rm MSY} = 1/2$.
Diffusion creates a **spillover gradient** near $\xi = 1/3$ where fish migrate
from the higher-density EEZ into the depleted international zone.

### 3C(ii) — Attractor Robustness (Fish Origin Outside the EEZ)

Same Scenario C harvest policy but initial Gaussian centred at $\xi_0 = 2/3$
(international waters). Long-run equilibrium matches 3C(i), confirming the
steady state is an **attractor independent of initial conditions**.

### Diagnostics (Dimensionless)

| Quantity | Definition |
|----------|------------|
| $B_{\rm in}(\tau)$ | $\int_0^{1/3} x\,d\xi$ — inshore dimensionless biomass |
| $B_{\rm out}(\tau)$ | $\int_{1/3}^{1} x\,d\xi$ — offshore dimensionless biomass |
| $B_{\rm tot}(\tau)$ | $\int_0^{1} x\,d\xi$ — total dimensionless biomass |
| Catch rate | $\int_0^{1} \tilde{h}(\xi,\tau)\,x\,d\xi$ — dimensionless instantaneous catch |
| Cumulative catch | $\int_0^{\tau} \text{Catch}\,d\tau'$ |

### Plots Per Scenario

1. **Density snapshots** — $x(\xi)$ profiles at selected $\tau$ with the $\xi=1/3$ boundary marked.
2. **Biomass time series** — $B_{\rm in}$, $B_{\rm out}$, and $B_{\rm tot}$ vs $\tau$.
3. **Catch plot** — dimensionless catch rate (left axis) and cumulative catch (right axis) vs $\tau$.

### 3D Surface and Heatmap Visualisations

For each scenario, the full space-time density field $x(\xi, \tau)$ is visualised:

- **3D surface plot** (matplotlib) — $\xi$ vs $\tau$ vs $x$.
- **Interactive 3D surface** (Plotly, if installed) — rotatable version.
- **Heatmap** — colour map of $x(\xi, \tau)$ with the $\xi = 1/3$ boundary overlaid.

Surface data is downsampled to a maximum of 200 points per axis for performance.

---

## 3D — Stochastic Annual Harvest Schedules

### Motivation

Harvest quotas fluctuate annually. Section 3D (implemented within 3C) replaces
fixed $\tilde{h}_{\rm in}$/$\tilde{h}_{\rm out}$ constants with **pre-generated
annual schedules** of dimensionless rates $\tilde{h}$.

### Implementation

Two 1-D arrays are passed to `simulate_rd_fishing` (as dimensional rates
$h = \tilde{h} \cdot r$; conversion to ND is applied after simulation):

```
h_in_schedule   shape (n_years,)   inshore  harvest rate for each year
h_out_schedule  shape (n_years,)   offshore harvest rate for each year
```

The active rates are looked up by integer year:

```
year = min(floor(t), n_years - 1)
```

### Stability

Time-step uses the maximum rate across the full schedule:

$$
\Delta t \le \frac{0.2}{\max\!\bigl(r,\;\max(\mathbf{h}_{\rm in}),\;\max(\mathbf{h}_{\rm out})\bigr)}
$$

---

## Spatial Domain

The 1-D dimensionless coordinate $\xi = s/L \in [0, 1]$ represents offshore position:

- $\xi = 0$ — coastline (shore boundary)
- $\xi = 1$ — far offshore boundary
- $\xi_{\rm bnd} = 1/3$ — 200-mile EEZ limit

Boundary conditions are **no-flux (Neumann)** at both ends: $\partial x/\partial\xi = 0$,
meaning no fish leave the domain. Implemented via ghost-point equivalence in
the finite-difference Laplacian.

---

## Numerical Methods Summary

| Method | Where Used | Details |
|--------|-----------|---------|
| RK45 (adaptive Runge-Kutta) | Sections 1, 2 (ODEs) | `scipy.integrate.solve_ivp`, `rtol=1e-9`, `atol=1e-12`; called with `r=1, K=1` for ND output |
| Explicit Euler + finite differences | Sections 3A, 3B, 3C (PDEs) | Second-order central differences in $\xi$, first-order forward Euler in $\tau$ |

**Stability (ND):** The explicit Euler PDE solver enforces
$\Delta\tau \le 0.45 \cdot \Delta\xi^2/(2\delta)$. Section 3C adds a
reaction-rate constraint in ND to prevent instability from harvesting terms.

**ND conversion for 3C:** `simulate_rd_fishing` is called with dimensional
parameters internally (so integer-year schedule indexing works correctly).
Results are converted to ND after each simulation via the `to_nd` helper
($\xi = s/L$, $\tau = rt$, $x = u/K$, biomass divided by $KL$, catch by $rKL$).

### `simulate_rd_fishing` Solver Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L`, `N`, `D`, `r`, `K` | — | Dimensional grid and physics parameters |
| `h_in` / `h_out` | `0.0` / `0.2` | Constant inshore/offshore harvest rates (dimensional) |
| `h_in_schedule` / `h_out_schedule` | `None` | Annual dimensional harvest-rate arrays `(n_years,)` |
| `pulse` | `None` | Dict `{t_start, t_end, h_out_pulse}` — temporary offshore fishing episode |
| `store_full` | `False` | If `True`, save full space-time field for 3D/heatmap plots |
| `full_n_frames` | `300` | Maximum time frames stored when `store_full=True` |
| `dt_safety` | `0.45` | Safety factor for the diffusion stability limit |
| `u0_type` | `"gaussian"` | IC type: `"gaussian"` or `"uniform"` |
| `gaussian_center` | `150.0` | Dimensional offshore distance of Gaussian IC peak ($= \xi_0 \cdot L$) |
| `gaussian_scale` | `50.0` | Width parameter $\sigma$ of the Gaussian IC (dimensional) |
| `snapshot_times` | `[0,10,20,40,60]` | Dimensional times for spatial snapshots |

---

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- SciPy
- Plotly (optional — interactive 3D surfaces; static matplotlib fallbacks shown if missing)

## Project Files

```
visualsv3.ipynb          Main notebook (orchestration + visualisation)
ode_models.py            Logistic & harvested ODE functions
pde_solver.py            Spatial discretisation & reaction-diffusion PDE solver
validation.py            Numerical validation utilities
plotting.py              All reusable plotting functions (labels in ND notation)
requirements.txt         Python dependencies
README.md                This file
.gitignore               Git ignore rules
```
