"""Numerical validation utilities for ODE solvers."""

from __future__ import annotations

import numpy as np

from ode_models import logistic_analytic, logistic_numeric


def validate_case(
    r: float,
    K: float,
    x0: float,
    t_eval: np.ndarray,
    negativity_tol: float = -1e-12,
) -> dict[str, float | bool]:
    """Validate one parameter case and return error/positivity metrics."""
    x_numeric = logistic_numeric(t_eval=t_eval, r=r, K=K, x0=x0)
    x_analytic = logistic_analytic(t=t_eval, r=r, K=K, x0=x0)

    max_abs_error = float(np.max(np.abs(x_numeric - x_analytic)))
    min_numeric = float(np.min(x_numeric))
    min_analytic = float(np.min(x_analytic))

    numeric_negative = bool(np.any(x_numeric < negativity_tol))
    analytic_negative = bool(np.any(x_analytic < negativity_tol))

    return {
        "r": r,
        "K": K,
        "x0": x0,
        "max_abs_error": max_abs_error,
        "min_numeric": min_numeric,
        "min_analytic": min_analytic,
        "numeric_negative": numeric_negative,
        "analytic_negative": analytic_negative,
    }


def run_validation_suite(
    cases: list[tuple[float, float, float]],
    t_eval: np.ndarray,
    negativity_tol: float = -1e-12,
) -> list[dict[str, float | bool]]:
    """Run validation over multiple (r, K, x0) sets."""
    results: list[dict[str, float | bool]] = []
    for r, K, x0 in cases:
        metrics = validate_case(r=r, K=K, x0=x0, t_eval=t_eval, negativity_tol=negativity_tol)
        results.append(metrics)

        if bool(metrics["numeric_negative"]):
            print(
                f"WARNING: numerical solution went negative for r={r:.3g}, K={K:.3g}, x0={x0:.3g} "
                f"(min={metrics['min_numeric']:.3e})."
            )
        if bool(metrics["analytic_negative"]):
            print(
                f"WARNING: analytic solution went negative for r={r:.3g}, K={K:.3g}, x0={x0:.3g} "
                f"(min={metrics['min_analytic']:.3e})."
            )

    return results
