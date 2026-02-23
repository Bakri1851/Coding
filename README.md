# Fish Population Modelling — Spatial Reaction-Diffusion with Harvesting

Numerical modelling of fish population dynamics, progressing from time-only ODEs
to a **1D spatial** (offshore distance only) reaction-diffusion PDE with spatially
varying harvesting, and finally to a **two-species Lotka-Volterra competition**
model. Each section validates against known solutions before the next layer is added.

All results are presented in **dimensionless form** (see
[Non-Dimensionalisation](#non-dimensionalisation) below).

## Notebook

**`visualsv3.ipynb`** is the primary notebook. It imports from companion modules
`ode_models.py`, `pde_solver.py`, `validation.py`, and `plotting.py`.
Run top-to-bottom (`Restart & Run All`) — no external data files are required.

---

## Models

| Section | Model | Spatial dimensions |
|---------|-------|--------------------|
| 1 | Logistic ODE | None (time only) |
| 2 | Logistic ODE with constant harvesting | None (time only) |
| 3A | Pure diffusion (migration only) | 1D — offshore distance $s$ |
| 3B | Logistic reaction-diffusion | 1D — offshore distance $s$ |
| 3C | Reaction-diffusion with EEZ fishing policy | 1D — offshore distance $s$ |
| 3D | Stochastic annual harvest schedules | 1D — offshore distance $s$ |
| 4 | Two-species Lotka-Volterra competition + diffusion | 1D — offshore distance $s$ |

> **Note on heatmaps:** The 3D surface plots and heatmaps in Section 3C display
> $x(\xi, \tau)$ as a colour map with offshore position $\xi$ on one axis and
> time $\tau$ on the other. These are visualisations of the **1D PDE solution**,
> not true 2D spatial PDEs (there is no diffusion in a second spatial direction
> in Sections 1–4).

---

## Non-Dimensionalisation

**Notation:**
- $N(t)$ — dimensional population size or biomass (individuals or kg).
- $x(\tau)$ — dimensionless population fraction $x = N/K \in [0,1]$.
- $u(s,t)$ — dimensional fish density (biomass per unit offshore distance), where
  $s \in [0, L]$ is the offshore distance and $L = 600\,\text{km}$ is the domain length.
- $x(\xi,\tau)$ — dimensionless density $x = u/K$ at dimensionless position
  $\xi = s/L \in [0,1]$.

| Substitution | Definition | Interpretation |
|---|---|---|
| $x = N/K$ (ODE) or $x = u/K$ (PDE) | Density scaled by carrying capacity | Population as fraction of $K$ |
| $\xi = s/L$ | Offshore position scaled by domain length | $\xi \in [0,1]$ |
| $\tau = rt$ | Time scaled by growth rate $r$ | Dimensionless time |
| $\gamma = h/r$ | Harvest rate scaled by growth rate | Ratio of harvesting to growth |
| $\delta = D/(rL^2)$ | Diffusion coefficient scaled by growth and domain area | Ratio of mixing to growth |

**Interpretation of key parameters:**
- $\gamma$: fraction of net production removed by fishing; $\gamma < 1$ is sustainable.
- $\delta$: compares the diffusion timescale $L^2/D$ to the growth timescale $1/r$.
  Small $\delta \ll 1$ means growth dominates mixing and spatial gradients persist;
  large $\delta \gg 1$ means rapid mixing drives the solution toward spatial homogeneity.

With base parameters ($D = 10$, $r = 0.5$, $L = 600$):
$$\delta = \frac{D}{rL^2} = \frac{10}{0.5 \times 360000} \approx 5.56 \times 10^{-5}$$

---

## 1 — Logistic Fish Population Model (Time-Only ODE)

### Dimensional model

$$\frac{dN}{dt} = r\,N\!\left(1 - \frac{N}{K}\right)$$

### Dimensionless form

Substituting $N = Kx$ and $t = \tau/r$ gives $rK\,dx/d\tau = rKx(1-x)$, so:

$$\frac{dx}{d\tau} = x(1 - x)$$

This ODE has **no free parameters** — $r$ and $K$ only enter via the initial
condition $x_0 = N_0/K$ and the timescale $\tau = rt$.

### Analytic solution

$$x(\tau) = \frac{1}{1 + \left(\dfrac{1 - x_0}{x_0}\right) e^{-\tau}}$$

Equilibria: $x^* = 0$ (unstable) and $x^* = 1$ (globally attracting for $x_0 > 0$).

### Numerical method

RK45 adaptive Runge-Kutta via `scipy.integrate.solve_ivp` (`rtol=1e-9`,
`atol=1e-12`), called with $r=1$, $K=1$ so the output is directly $x(\tau)$.

### Validation

Four dimensional cases $(r, K, N_0)$ are reduced to $x_0 = N_0/K$ and run on a
common $\tau$-grid to $\tau = 15$. Maximum absolute error against the analytic
solution and positivity are reported.

| Case | $r$ | $K$ | $N_0$ | $x_0 = N_0/K$ |
|------|-----|-----|--------|----------------|
| 1 | 0.3 | 500 | 50 | 0.10 |
| 2 | 0.8 | 1000 | 100 | 0.10 |
| 3 | 1.2 | 1200 | 1500 | 1.25 |
| 4 | 0.5 | 300 | 290 | 0.967 |

Cases 1 and 2 have identical ND dynamics ($x_0 = 0.10$), confirming that $r$ and $K$
individually are irrelevant once the solution is scaled.

### Outputs

- $x(\tau)$ for $x_0 = 0.1$, $\tau \in [0, 18]$.
- RK45 vs analytic comparison for $x_0 \in \{0.1, 0.5, 1.25\}$.

---

## 2 — Logistic Growth with Harvesting

### Dimensional model

$$\frac{dN}{dt} = r\,N\!\left(1 - \frac{N}{K}\right) - h\,N$$

where $h \ge 0$ is a constant proportional harvesting rate.

### Dimensionless derivation

Substitute $N = Kx$ and $t = \tau/r$. The equation becomes:

$$rK\,\frac{dx}{d\tau} = r\,Kx(1-x) - h\,Kx$$

Dividing through by $rK$ and writing $\gamma = h/r$:

$$\frac{dx}{d\tau} = x(1 - x) - \gamma\,x = x\!\left(1 - x - \gamma\right)$$

The harvest term is $-\gamma x$, not $-\gamma$: it removes a fraction $\gamma$ of
the current population, not a fixed amount.

### Analytic behaviour

| Regime | Condition | Non-zero equilibrium $x^*$ |
|--------|-----------|----------------------------|
| Sustainable | $\gamma < 1$ | $1 - \gamma$ |
| Critical | $\gamma = 1$ | $0$ (marginal collapse) |
| Overfishing | $\gamma > 1$ | none — extinction ($x \to 0$) |

**Maximum Sustainable Yield:** $\gamma_{\rm MSY} = 1/2$, achieved at $x^*_{\rm MSY} = 1/2$.

### Parameters tested

Base: $r = 0.5$, $K = 1000$, $N_0 = 500$ ($x_0 = 0.5$), $t \in [0,40]$ ($\tau \in [0,20]$).

| $h$ | $\gamma = h/r$ | Equilibrium $x^* = 1 - \gamma$ | Outcome |
|-----|----------------|-------------------------------|---------|
| 0.00 | 0.00 | 1.00 | No fishing |
| 0.10 | 0.20 | 0.80 | Sustainable |
| 0.30 | 0.60 | 0.40 | Moderate depletion |
| 0.45 | 0.90 | 0.10 | Near collapse |
| 0.55 | 1.10 | — | Extinction |

### Outputs

- $x(\tau)$ trajectories for each $\gamma$ value; dashed lines mark analytic
  equilibria $x^* = 1 - \gamma$.

---

## 3A — Pure Diffusion Sanity Check (Migration Only)

### Governing equation (dimensionless)

$$\frac{\partial x}{\partial \tau} = \delta\,\frac{\partial^2 x}{\partial \xi^2}, \qquad
\frac{\partial x}{\partial \xi}\bigg|_{\xi=0} = \frac{\partial x}{\partial \xi}\bigg|_{\xi=1} = 0$$

Here $x(\xi,\tau) = u(s,t)/K$ is the dimensionless fish density, $\xi = s/L$,
$\tau = r_{\rm ref}\,t$ (with reference $r_{\rm ref} = 0.5$).

**Purpose:** validate the finite-difference Laplacian and Neumann boundary
conditions in isolation, before adding reaction or harvesting terms.

Expected behaviour: an initial Gaussian (centred at $\xi = 0.25$) spreads and
flattens; total mass $\int_0^1 x\,d\xi$ is conserved to machine precision
(no flux out, no reaction).

### Grid and parameters

| Parameter | Value |
|-----------|-------|
| $\delta$ | $\approx 5.56 \times 10^{-5}$ |
| $N$ (grid points) | 301 |
| $\Delta\xi$ | $\approx 3.33 \times 10^{-3}$ |
| $\tau_{\rm end}$ | 10.0 |
| IC | $x(\xi,0) = \exp\!\bigl(-\bigl((\xi - 0.25)/0.0833\bigr)^2\bigr)$ |

### Finite-difference Laplacian

Interior points (second-order central difference):
$$\frac{\partial^2 x}{\partial \xi^2}\bigg|_i \approx \frac{x_{i+1} - 2x_i + x_{i-1}}{\Delta\xi^2}$$

Boundary points (ghost-point equivalence for Neumann conditions):
$$\frac{\partial^2 x}{\partial \xi^2}\bigg|_0 \approx \frac{2(x_1 - x_0)}{\Delta\xi^2}, \qquad
\frac{\partial^2 x}{\partial \xi^2}\bigg|_{N-1} \approx \frac{2(x_{N-2} - x_{N-1})}{\Delta\xi^2}$$

Time-step stability: $\Delta\tau \le 0.45 \cdot \Delta\xi^2/(2\delta)$.

### Outputs

- Density snapshots at $\tau = 0, 2.5, 5, 10$.
- Mass conservation plot confirming $\int_0^1 x\,d\xi = \text{const}$.

---

## 3B — Reaction-Diffusion: Logistic Growth + Migration

### Dimensional model

$$\frac{\partial u}{\partial t} = r\,u\!\left(1 - \frac{u}{K}\right) + D\,\frac{\partial^2 u}{\partial s^2}$$

where $u(s,t)$ is fish density (biomass per unit offshore distance), $s \in [0,L]$.

### Dimensionless form

Substituting $x = u/K$, $\xi = s/L$, $\tau = rt$:

$$\frac{\partial x}{\partial \tau} = x(1-x) + \delta\,\frac{\partial^2 x}{\partial \xi^2}, \qquad
\frac{\partial x}{\partial \xi}\bigg|_{\xi=0} = \frac{\partial x}{\partial \xi}\bigg|_{\xi=1} = 0$$

where $\delta = D/(rL^2)$ (mixing-to-growth ratio; see
[Non-Dimensionalisation](#non-dimensionalisation)).

### Validation

A uniform IC $x(\xi,0) = 0.2$ satisfies $\partial^2 x/\partial\xi^2 = 0$ for all
time, so the PDE reduces to $dx/d\tau = x(1-x)$. The PDE spatial mean is overlaid
against the analytic logistic solution; maximum spatial deviation remains at
machine epsilon.

### Parameters

| Parameter | Dimensional | Dimensionless |
|-----------|-------------|---------------|
| $r$ | 0.5 | — |
| $K$ | 1.0 | — |
| $D$ | 10.0 | $\delta \approx 5.56 \times 10^{-5}$ |
| $T$ | 60.0 | $\tau_{\rm end} = 30.0$ |

### Outputs

- Density snapshots $x(\xi)$ at $\tau = 0, 5, 10, 20, 30$ for a Gaussian IC.
- Spatial mean vs analytic logistic solution for a uniform IC.

---

## 3C — Reaction-Diffusion with Fishing Policy (200-Mile EEZ)

### Dimensional model

$$\frac{\partial u}{\partial t} = r\,u\!\left(1 - \frac{u}{K}\right) - h(s)\,u + D\,\frac{\partial^2 u}{\partial s^2}$$

### Dimensionless form

$$\frac{\partial x}{\partial \tau} = x(1-x) - \gamma(\xi)\,x + \delta\,\frac{\partial^2 x}{\partial \xi^2}$$

The dimensionless harvest rate $\gamma(\xi) = h(s)/r$ is piecewise constant across
the 200-mile EEZ boundary at $\xi_{\rm bnd} = 200/600 = 1/3$:

$$\gamma(\xi) = \begin{cases}
\gamma_{\rm in}  & \xi \le 1/3 \quad \text{(EEZ — regulated)} \\[4pt]
\gamma_{\rm out} & \xi > 1/3 \quad \text{(international waters)}
\end{cases}$$

### Harvest calibration

$$\gamma_{\rm MSY} = \tfrac{1}{2}, \qquad
\gamma_{\rm MEY} = \tfrac{1}{2}\!\left(1 - \sqrt{1-f}\right), \quad f \sim \mathrm{Uniform}(0.80, 0.90)$$

### Three policy scenarios

| Scenario | $\gamma_{\rm in}$ (EEZ) | $\gamma_{\rm out}$ (international) | Long-run outcome |
|----------|------------------------|------------------------------------|------------------|
| **A** | 0 | 0 | $x^* = 1$ everywhere (no fishing) |
| **B** | $\gamma_{\rm MEY}$ | $\gamma_{\rm MEY}$ | $x^* = 1 - \gamma_{\rm MEY}$ uniformly |
| **C** | $\gamma_{\rm MEY}$ | $\gamma_{\rm MSY} = 1/2$ | Spillover gradient near $\xi = 1/3$ |

In Scenario C, higher-density EEZ fish diffuse into the more heavily fished
international zone, creating a **spillover gradient** near the boundary.

### Outputs per scenario

1. **Density snapshots** — $x(\xi)$ profiles at selected $\tau$; $\xi = 1/3$ boundary marked.
2. **Biomass time series** — $B_{\rm in}(\tau) = \int_0^{1/3} x\,d\xi$,
   $B_{\rm out}(\tau) = \int_{1/3}^{1} x\,d\xi$, $B_{\rm tot}(\tau) = \int_0^{1} x\,d\xi$.
3. **Catch plot** — dimensionless catch rate $\int_0^1 \gamma(\xi)\,x\,d\xi$ and cumulative catch vs $\tau$.
4. **Space-time heatmap** — colour map of $x(\xi,\tau)$ with $\xi$ on the horizontal axis
   and $\tau$ on the vertical axis. This is a visualisation of the 1D PDE solution plotted
   over time; it is **not** a 2D spatial PDE.

### 3C(ii) — Attractor robustness

Same Scenario C policy but with the Gaussian IC centred at $\xi_0 = 2/3$
(international waters). The long-run equilibrium matches 3C(i), confirming the
steady state is an attractor independent of initial conditions.

---

## 3D — Stochastic Annual Harvest Schedules

Extends Section 3C by replacing fixed $\gamma_{\rm in}$/$\gamma_{\rm out}$ with
**pre-generated annual arrays** of dimensionless rates. The active rate for year
$n$ is selected by $n = \lfloor t \rfloor$ (with $t$ in years). The time-step
stability condition uses the maximum rate across the full schedule.

---

## 4 — Two-Species Competing Reaction-Diffusion with Harvesting

### Dimensional model (Lotka-Volterra competition)

$$\frac{dN_1}{dt} = r_1 N_1\!\left(1 - \frac{N_1 + \alpha_{12}\,N_2}{K_1}\right) - h_1\,N_1$$

$$\frac{dN_2}{dt} = r_2 N_2\!\left(1 - \frac{N_2 + \alpha_{21}\,N_1}{K_2}\right) - h_2\,N_2$$

where $\alpha_{12}$ is the competitive effect of species 2 on species 1, and
$\alpha_{21}$ is the competitive effect of species 1 on species 2.
The **PDE extension** adds diffusion terms $D_i\,\partial^2 N_i/\partial s^2$ and
the same piecewise EEZ harvesting as Section 3C, operating on the same 1D
offshore domain $s \in [0, L]$.

### Dimensionless PDEs

With $u = N_1/K_1$, $v = N_2/K_2$, $\xi = s/L$, $\tau = r_1 t$:

$$\frac{\partial u}{\partial \tau} = u\!\left(1 - u - \alpha_{12}\,\frac{K_2}{K_1}\,v\right) - \gamma_1(\xi)\,u + \delta_1\,\frac{\partial^2 u}{\partial \xi^2}$$

$$\frac{\partial v}{\partial \tau} = \frac{r_2}{r_1}\,v\!\left(1 - v - \alpha_{21}\,\frac{K_1}{K_2}\,u\right) - \gamma_2(\xi)\,v + \delta_2\,\frac{\partial^2 v}{\partial \xi^2}$$

where $\delta_i = D_i/(r_1 L^2)$, $\gamma_i = h_i/r_1$, and no-flux Neumann
conditions apply at both boundaries for each species.

**Coexistence condition:** stable coexistence requires $\alpha_{12}\alpha_{21} < 1$;
competitive exclusion occurs when $\alpha_{12}\alpha_{21} > 1$.

### Outputs

- Density snapshots for each species at selected times (species 1 solid, species 2 dashed).
- Biomass time series for both species.
- Catch rate and cumulative catch per species.
- Side-by-side space-time heatmaps of $u(\xi,\tau)$ and $v(\xi,\tau)$
  (1D PDE visualisations over time, not 2D spatial PDEs).

---

## Spatial Domain

The 1D dimensionless domain $\xi = s/L \in [0,1]$ represents offshore distance:

- $\xi = 0$ — coastline
- $\xi = 1$ — far offshore boundary ($s = L = 600\,\text{km}$)
- $\xi_{\rm bnd} = 1/3$ — 200-mile EEZ limit ($s = 200\,\text{km}$)

**Boundary conditions:** no-flux (Neumann) at both ends, $\partial x/\partial\xi = 0$,
implemented via ghost-point finite differences (see Section 3A).

---

## Numerics

| Method | Sections | Details |
|--------|----------|---------|
| RK45 (adaptive) | 1, 2 | `scipy.integrate.solve_ivp`, `rtol=1e-9`, `atol=1e-12`; run with $r=1$, $K=1$ for direct ND output |
| Explicit Euler + central differences | 3A–4 | Second-order FD in $\xi$; first-order forward Euler in $\tau$ |

**Stability conditions (dimensionless):**

$$\Delta\tau \le \frac{0.45\,\Delta\xi^2}{2\delta} \qquad \text{(diffusion limit)}$$

$$\Delta\tau \le \frac{0.2}{\max(1,\,\gamma_{\rm max})} \qquad \text{(reaction/harvest limit)}$$

The actual $\Delta\tau$ is the minimum of both constraints; for the two-species model
the minimum is taken across both species. The PDE solver uses dimensional inputs
internally; results are converted to ND after simulation via $x = u/K$,
$\xi = s/L$, $\tau = rt$.

---

## How to Run

All sections run from **`visualsv3.ipynb`**. Open the notebook and execute
`Restart & Run All`. Sections can also be run individually in order.

| Section | Key outputs |
|---------|-------------|
| §1 | Logistic ODE trajectories and analytic comparison plots |
| §2 | Harvested ODE trajectories for multiple $\gamma$ values |
| §3A | Pure-diffusion Gaussian spreading + mass conservation check |
| §3B | Reaction-diffusion density snapshots + logistic validation |
| §3C | EEZ fishing policy scenarios — density snapshots, biomass time series, catch plots, space-time heatmaps |
| §3D | Stochastic annual harvest schedule trajectories |
| §4 | Two-species competition — density snapshots, biomass time series, catch plots, heatmaps |

The companion modules (`ode_models.py`, `pde_solver.py`, `validation.py`,
`plotting.py`) must reside in the same directory as the notebook.

---

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- SciPy
- Plotly (optional — interactive 3D surfaces in Section 3C; static Matplotlib
  fallbacks are shown if Plotly is not installed)

## Project Files

```
visualsv3.ipynb          Main notebook (orchestration + visualisation)
ode_models.py            Logistic and harvested ODE functions
pde_solver.py            Spatial discretisation and reaction-diffusion PDE solver
validation.py            Numerical validation utilities
plotting.py              All reusable plotting functions (labels in ND notation)
requirements.txt         Python dependencies
README.md                This file
.gitignore               Git ignore rules
```
