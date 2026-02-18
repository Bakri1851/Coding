# Fish Population Modelling — Spatial Reaction-Diffusion with Harvesting

Numerical modelling of single-species fish population dynamics, progressing from
time-only ODEs to a 1-D spatial reaction-diffusion PDE with spatially varying
harvesting. The project builds complexity incrementally: each section validates
against known solutions or limiting cases before the next layer is added.

## Notebook

The primary notebook is **`visualsv3.ipynb`**. It imports model, solver,
validation, and plotting functions from the companion Python modules
(`ode_models.py`, `pde_solver.py`, `validation.py`, `plotting.py`).
Run top-to-bottom (`Restart & Run All`) with no external data files.

---

## 1 — Logistic Fish Population Model (Time-Only ODE)

### Governing Equation

$$
\frac{dx}{dt} = r\,x\!\left(1 - \frac{x}{K}\right), \qquad r > 0,\; K > 0,\; x(0) = x_0 > 0
$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $x(t)$ | Population at time $t$ | — |
| $r$ | Intrinsic growth rate | 0.6 |
| $K$ | Carrying capacity | 1000 |
| $x_0$ | Initial population | varies |

### Analytic Solution

$$
x(t) = \frac{K}{1 + \left(\frac{K - x_0}{x_0}\right) e^{-rt}}
$$

Equilibria: $x^* = 0$ (unstable) and $x^* = K$ (globally attracting for $x_0 > 0$).

### Numerical Method

RK45 adaptive Runge-Kutta via `scipy.integrate.solve_ivp` with `rtol=1e-9`,
`atol=1e-12`.

### Validation

- **Parameter sweep:** Four $(r, K, x_0)$ cases are integrated numerically and
  compared against the closed-form solution. Maximum absolute errors are
  reported and positivity is checked (threshold $-10^{-12}$).
- **Initial-condition comparison:** Three initial conditions
  ($0.1K$, $0.5K$, $1.2K$) all converge to $K$, confirming global stability.
- **Growth-rate sweep:** $r \in \{0.2, 0.5, 1.0, 1.5\}$ with fixed $x_0 = 0.1K$
  shows faster convergence for larger $r$ while all trajectories reach $K$.

| Validation case | $r$ | $K$ | $x_0$ |
|-----------------|-----|-----|--------|
| 1 | 0.3 | 500 | 50 |
| 2 | 0.8 | 1000 | 100 |
| 3 | 1.2 | 1200 | 1500 |
| 4 | 0.5 | 300 | 290 |

### Plots

| Figure | Description |
|--------|-------------|
| Quick-look logistic curve | Single trajectory ($x_0=100$) rising to $K=1000$ |
| Initial-condition comparison | RK45 vs analytic for $x_0 \in \{100, 500, 1200\}$ |
| Growth-rate sweep | $r$-dependence of convergence speed |

---

## 2 — Logistic Growth with Fishing (Harvesting)

### Governing Equation

$$
\frac{dx}{dt} = r\,x\!\left(1 - \frac{x}{K}\right) - h\,x
$$

Proportional harvesting at rate $h$ removes biomass in proportion to the
current population.

### Analytic Behaviour

| Regime | Condition | Effective equilibrium |
|--------|-----------|----------------------|
| Sustainable | $h < r$ | $x^* = K\!\left(1 - \frac{h}{r}\right)$ |
| Critical | $h = r$ | $x^* = 0$ (marginal collapse) |
| Overfishing | $h > r$ | $x \to 0$ (extinction) |

When $h < r$ the ODE reduces to a logistic equation with effective growth rate
$a = r - h$ and effective carrying capacity $K_{\mathrm{eff}} = K(1 - h/r)$,
so the analytic solution is:

$$
x(t) = \frac{K_{\mathrm{eff}}}{1 + \left(\frac{K_{\mathrm{eff}} - x_0}{x_0}\right) e^{-a\,t}}
$$

### Parameters

| Symbol | Value |
|--------|-------|
| $r$ | 0.50 |
| $K$ | 1000 |
| $x_0$ | 500 |
| $h$ values tested | 0.00, 0.10, 0.30, 0.45, 0.55 |
| Time span | $[0, 40]$ |

### Key Results

- $h = 0.00$: population rises to $K = 1000$.
- $h = 0.10$: equilibrium at $K(1 - 0.10/0.50) = 800$.
- $h = 0.30$: equilibrium at $K(1 - 0.30/0.50) = 400$.
- $h = 0.45$: equilibrium at $K(1 - 0.45/0.50) = 100$.
- $h = 0.55 > r$: overfishing — population collapses toward zero.

Dashed horizontal lines on the plot mark each analytic equilibrium.

---

## 3A — Pure Diffusion Sanity Check (Migration Only)

### Governing Equation

$$
\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial s^2}
$$

with no-flux (Neumann) boundary conditions at both ends:

$$
\frac{\partial u}{\partial s}(0, t) = 0, \qquad \frac{\partial u}{\partial s}(L, t) = 0
$$

### Purpose

Validate the spatial finite-difference discretisation and boundary treatment
**in isolation** before adding reaction or fishing terms. The expected
behaviour is:

1. An initial Gaussian bump (centred at $s = 150$) spreads and flattens.
2. Total mass $M(t) = \int_0^L u\,ds$ is conserved (no flux out, no reaction).

### Spatial Grid and Parameters

| Parameter | Value |
|-----------|-------|
| Domain $s \in [0, L]$ | $L = 600$ (offshore distance) |
| Grid points $N$ | 301 |
| Grid spacing $\Delta s$ | $L/(N-1) = 2.0$ |
| Diffusion coefficient $D$ | 10.0 |
| End time $T$ | 20.0 |
| Initial condition | $u_0(s) = \exp\!\bigl(-((s - 150)/50)^2\bigr)$ (peak at $s = 150$) |

### Finite-Difference Laplacian

Interior points use the standard second-order central difference:

$$
\frac{\partial^2 u}{\partial s^2}\bigg|_i \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta s^2}
$$

At the boundaries, the **ghost-point equivalence** for Neumann conditions gives:

$$
\frac{\partial^2 u}{\partial s^2}\bigg|_0 \approx \frac{2(u_1 - u_0)}{\Delta s^2}, \qquad
\frac{\partial^2 u}{\partial s^2}\bigg|_{N-1} \approx \frac{2(u_{N-2} - u_{N-1})}{\Delta s^2}
$$

### Time Stepping

Explicit Euler with a diffusion-stability-guided time step:

$$
\Delta t \le \frac{\Delta s^2}{2D}
$$

A safety factor of 0.45 is applied.

### Validation

- **Density snapshots** at $t = 0, 5, 10, 20$ show the Gaussian spreading.
- **Mass conservation plot** confirms $M(t)$ remains constant to machine precision.

---

## 3B — Reaction-Diffusion: Logistic Growth + Migration (No Fishing)

### Governing Equation

$$
\frac{\partial u}{\partial t} = r\,u\!\left(1 - \frac{u}{K}\right) + D\,\frac{\partial^2 u}{\partial s^2}
$$

Same domain, grid, boundary conditions, and numerical method as 3A.

### Parameters

| Parameter | Value |
|-----------|-------|
| $r$ | 0.5 |
| $K$ | 1.0 |
| $D$ | 10.0 |
| $T$ | 60.0 |

### Sub-experiments

#### Part A — Non-uniform Initial Condition (Gaussian Bump)

$u_0(s) = \exp\!\bigl(-((s - 150)/50)^2\bigr)$

**Behaviour:** The bump peaks at $s = 150$ (mid-inshore), with density falling
off toward both the coastline and far offshore. Logistic growth fills the
low-density tails while diffusion smooths spatial gradients. The profile
converges uniformly to $K$.

Snapshots are shown at $t = 0, 10, 20, 40, 60$.

#### Part B — Uniform Initial Condition Validation

$u_0(s) = 0.2K$ everywhere.

**Purpose:** When the initial condition is spatially uniform, $\partial^2 u/\partial s^2 = 0$
everywhere for all time, so the PDE should reduce exactly to the logistic ODE
$du/dt = ru(1 - u/K)$. The PDE spatial mean is overlaid against the analytic
logistic solution; the maximum spatial deviation remains at machine epsilon,
confirming the discretisation introduces no spurious spatial gradients.

---

## 3C — Reaction-Diffusion with Fishing Policy (200-Mile Boundary)

### Governing Equation

$$
\frac{\partial u}{\partial t} = r\,u\!\left(1 - \frac{u}{K}\right) - h(s,t)\,u + D\,\frac{\partial^2 u}{\partial s^2}
$$

The fishing rate is **piecewise constant** across a 200-mile economic-zone
boundary:

$$
h(s,t) = \begin{cases}
h_{\mathrm{in}}  & s \le 200 \\[4pt]
h_{\mathrm{out}} & s > 200
\end{cases}
$$

### Harvest Rate Calibration via MSY / MEY

Rather than choosing arbitrary harvest rates, we derive them from fisheries
economics. The **Maximum Sustainable Yield** for the logistic model is:

$$
\mathrm{MSY} = \frac{Kr}{4}, \qquad h_{\mathrm{MSY}} = \frac{r}{2}
$$

Fishing at MSY leaves no economic safety margin, so we target the **Maximum
Economic Yield (MEY)**, taken as a random fraction $f \in [0.80,\, 0.90]$ of
MSY. The corresponding per-capita harvest rate is:

$$
h_{\mathrm{MEY}} = \frac{r}{2}\left(1 - \sqrt{1 - f}\right)
$$

With $K = 1.0$, $r = 0.5$: $h_{\mathrm{MSY}} = 0.25$,
$h_{\mathrm{MEY}} \approx 0.138\text{–}0.171$ depending on the draw.

### Parameters

| Parameter | Value |
|-----------|-------|
| $L, N, D, r, K$ | Same as 3B |
| $s_{\mathrm{boundary}}$ | 200 |
| $T$ | 60.0 |

### Additional Stability Constraint

Because the harvesting term can be stiff, an additional time-step restriction
is imposed:

$$
\Delta t \le \frac{0.2}{\max(r,\, h_{\mathrm{in}},\, h_{\mathrm{out}})}
$$

The actual $\Delta t$ is the minimum of this and the diffusion stability limit.

### Three Policy Scenarios

| Scenario | $h_{\mathrm{in}}$ | $h_{\mathrm{out}}$ | Description |
|----------|-------------------|---------------------|-------------|
| **A** | 0.0 | 0.0 | No fishing (baseline, identical to 3B) |
| **B** | $h_{\mathrm{MEY}}$ | $h_{\mathrm{MEY}}$ | Uniform MEY fishing everywhere |
| **C** | 0.0 | $h_{\mathrm{MEY}}$ | 200-mile ban: inshore protected, MEY offshore |

### Key Results

**Scenario A (No Fishing):**
Growth fills the domain to $K$ from the initial Gaussian bump, identical to 3B.
Total biomass increases monotonically.

**Scenario B (Uniform MEY Fishing):**
The MEY harvest rate $h_{\mathrm{MEY}} < r$ depresses the equilibrium below $K$
everywhere. The effective carrying capacity is
$K(1 - h_{\mathrm{MEY}}/r)$, which the solution converges to uniformly. This
sustains near-optimal long-term yield while maintaining a safety margin below
MSY.

**Scenario C (200-Mile Ban + MEY Offshore):**
The inshore zone ($s \le 200$) is protected and reaches $K$. The offshore zone
is harvested at the MEY rate and converges to a lower equilibrium. Near the
boundary, diffusion creates a **spillover gradient**: fish migrate from the
high-density protected zone into the fished zone, partially replenishing
offshore biomass. This demonstrates how marine protected areas can sustain
adjacent fisheries.

### Diagnostics Tracked Per Scenario

| Quantity | Definition |
|----------|------------|
| $B_{\mathrm{in}}(t)$ | $\int_0^{s_{\mathrm{bnd}}} u\,ds$ — inshore biomass |
| $B_{\mathrm{out}}(t)$ | $\int_{s_{\mathrm{bnd}}}^{L} u\,ds$ — offshore biomass |
| $B_{\mathrm{tot}}(t)$ | $\int_0^{L} u\,ds$ — total biomass |
| Catch rate | $\int_0^{L} h(s,t)\,u\,ds$ — instantaneous catch |
| Cumulative catch | $\int_0^{t} \text{Catch}(\tau)\,d\tau$ |

### Plots Per Scenario

Each scenario generates three figure types:

1. **Density snapshots** — $u(s)$ profiles at $t = 0, 10, 20, 40, 60$ with the
   200-mile boundary marked.
2. **Biomass time series** — $B_{\mathrm{in}}$, $B_{\mathrm{out}}$, and
   $B_{\mathrm{tot}}$ vs time.
3. **Catch plot** — Instantaneous catch rate (left axis) and cumulative catch
   (right axis) vs time.

### 3D Surface and Heatmap Visualisations

For each scenario, the full space-time density field $u(s, t)$ is visualised:

- **3D surface plot** (matplotlib) — offshore distance $s$ vs time $t$ vs fish
  density $u$, rendered as a static plot that displays on GitHub.
- **Interactive 3D surface** (Plotly, if installed) — rotatable version for
  local exploration.
- **Heatmap** — colour map of $u(s, t)$ with the 200-mile policy boundary
  overlaid as a dashed red line.

Surface data is downsampled to a maximum of 200 points per axis for
performance.

---

## Spatial Domain

The 1-D coordinate $s$ represents **offshore distance**:

- $s = 0$ — coastline (shore boundary)
- $s = L = 600$ — far offshore boundary

Boundary conditions are **no-flux (Neumann)** at both ends: $\partial u/\partial s = 0$,
meaning no fish leave the domain. This is implemented via ghost-point
equivalence in the finite-difference Laplacian.

## Numerical Methods Summary

| Method | Where Used | Details |
|--------|-----------|---------|
| RK45 (adaptive Runge-Kutta) | Sections 1, 2 (ODEs) | `scipy.integrate.solve_ivp`, `rtol=1e-9`, `atol=1e-12` |
| Explicit Euler + finite differences | Sections 3A, 3B, 3C (PDEs) | Second-order central differences in space, first-order forward Euler in time |

**Stability:** The explicit Euler PDE solver enforces
$\Delta t \le 0.45 \cdot \Delta s^2/(2D)$ for diffusion stability. Section 3C
adds a reaction-rate constraint $\Delta t \le 0.2/\max(r, h_{\max})$ to
prevent instability from harvesting terms.

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- SciPy
- Plotly (optional — interactive 3D surfaces; static matplotlib fallbacks are
  shown if missing)

## Project Files

```
visualsv3.ipynb          Main notebook (orchestration + visualisation)
ode_models.py            Logistic & harvested ODE functions
pde_solver.py            Spatial discretisation & reaction-diffusion PDE solver
validation.py            Numerical validation utilities
plotting.py              All reusable plotting functions
requirements.txt         Python dependencies
README.md                This file
.gitignore               Git ignore rules
```
