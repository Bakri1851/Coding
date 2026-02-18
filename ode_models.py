"""Core ODE model functions for fish population dynamics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def logistic_rhs(t: float, x: np.ndarray, r: float, K: float) -> np.ndarray:
    """Return dx/dt for the logistic model."""
    del t  # Autonomous ODE: RHS does not explicitly depend on time.
    return r * x * (1.0 - x / K)


def logistic_analytic(t: np.ndarray, r: float, K: float, x0: float) -> np.ndarray:
    """Closed-form logistic solution for x0 > 0, r > 0, K > 0."""
    if r <= 0.0:
        raise ValueError(f"r must be positive, got {r}")
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")
    if x0 <= 0.0:
        raise ValueError(f"x0 must be positive for analytic formula, got {x0}")

    t_arr = np.asarray(t, dtype=float)
    factor = (K - x0) / x0
    return K / (1.0 + factor * np.exp(-r * t_arr))


def logistic_numeric(
    t_eval: np.ndarray,
    r: float,
    K: float,
    x0: float,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> np.ndarray:
    """Numerical logistic solution on a provided time grid using RK45."""
    if r <= 0.0:
        raise ValueError(f"r must be positive, got {r}")
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")
    if x0 <= 0.0:
        raise ValueError(f"x0 must be positive, got {x0}")

    t_arr = np.asarray(t_eval, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError("t_eval must be a 1D array with at least two points")

    sol = solve_ivp(
        fun=lambda t, y: logistic_rhs(t=t, x=y, r=r, K=K),
        t_span=(float(t_arr[0]), float(t_arr[-1])),
        y0=np.array([x0], dtype=float),
        t_eval=t_arr,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.y[0]


def harvested_rhs(t: float, x: np.ndarray, r: float, K: float, h: float) -> np.ndarray:
    """Right-hand side for harvested logistic growth."""
    del t  # Autonomous ODE.
    return r * x * (1.0 - x / K) - h * x


def harvested_analytic(
    t: np.ndarray,
    x0: float,
    r: float,
    K: float,
    h: float,
):
    """Analytic harvested logistic solution for h < r; otherwise None."""
    if h >= r:
        return None

    t_arr = np.asarray(t, dtype=float)
    a = r - h
    K_eff = K * (1.0 - h / r)
    return K_eff / (1.0 + ((K_eff - x0) / x0) * np.exp(-a * t_arr))
