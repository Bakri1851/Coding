"""2D reaction-diffusion solvers with periodic-x and Neumann-y boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np


def laplacian_2d_periodicx_neumanny(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return the 2D Laplacian with periodic x and no-flux y boundaries.

    Parameters
    ----------
    U
        Field array with shape (ny, nx), axis 0 = y and axis 1 = x.
    dx, dy
        Grid spacing in x and y.
    """
    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U.shape}.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    Ux_plus = np.roll(U, -1, axis=1)
    Ux_minus = np.roll(U, 1, axis=1)

    U_pad = np.pad(U, ((1, 1), (0, 0)), mode="edge")
    Uy_plus = U_pad[2:, :]
    Uy_minus = U_pad[:-2, :]

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    return (Ux_plus - 2.0 * U + Ux_minus) * inv_dx2 + (Uy_plus - 2.0 * U + Uy_minus) * inv_dy2


def fishing_profile_y(
    y_grid: np.ndarray,
    t: float,
    y_boundary: float,
    h_in: float,
    h_out: float,
    pulse: dict | None = None,
) -> np.ndarray:
    """Return y-dependent fishing profile h(y, t) with optional offshore pulse."""
    if y_grid.ndim != 1:
        raise ValueError("y_grid must be 1D.")

    h = np.where(y_grid <= y_boundary, h_in, h_out).astype(float)
    if pulse is None:
        return h

    t0 = pulse.get("t0", pulse.get("t_start"))
    t1 = pulse.get("t1", pulse.get("t_end"))
    h_out_pulse = pulse.get("h_out_pulse")
    if t0 is None or t1 is None or h_out_pulse is None:
        raise ValueError("pulse must define t0/t1 (or t_start/t_end) and h_out_pulse.")

    if t0 <= t <= t1:
        offshore = y_grid > y_boundary
        h[offshore] = float(h_out_pulse)
    return h


def simulate_single_2d(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    D: float,
    r: float,
    K: float,
    T_end: float,
    y_boundary: float = 200.0,
    h_in: float = 0.0,
    h_out: float = 0.0,
    pulse: dict | None = None,
    snapshot_times: list[float] | None = None,
    ic_mode: str = "noisy_baseline",
    noise_eps: float = 0.04,
    n_blobs: int = 7,
    seed: int | None = None,
    return_totals: bool = True,
    run_sanity_checks: bool = True,
    mass_tol: float = 1e-3,
    uniform_tol: float = 1e-10,
) -> dict[str, Any]:
    """Simulate 2D logistic growth + diffusion + y-dependent fishing.

    PDE:
        U_t = r*U*(1 - U/K) - h(y,t)*U + D*(U_xx + U_yy)
    """
    _validate_common_inputs(Lx=Lx, Ly=Ly, nx=nx, ny=ny, T_end=T_end)
    if D <= 0.0:
        raise ValueError("D must be positive for the explicit diffusion timestep rule.")
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if n_blobs < 1:
        raise ValueError("n_blobs must be >= 1.")

    x, y, dx, dy = _build_grid(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    dt_raw = 0.2 * min(dx * dx, dy * dy) / (4.0 * D)
    nt = int(np.ceil(T_end / dt_raw))
    dt = T_end / nt
    time = np.linspace(0.0, T_end, nt + 1)

    snap_targets, snap_steps = _snapshot_schedule(snapshot_times=snapshot_times, T_end=T_end, dt=dt, nt=nt)

    rng = np.random.default_rng(seed)
    U = _init_single_field(
        x=x,
        y=y,
        K=K,
        ic_mode=ic_mode,
        noise_eps=noise_eps,
        n_blobs=n_blobs,
        rng=rng,
    )
    min_u = float(U.min())

    snapshots: dict[float, np.ndarray] = {}
    if 0 in snap_steps:
        for t_req in snap_steps[0]:
            snapshots[t_req] = U.copy()

    diffusion_only = (r == 0.0 and h_in == 0.0 and h_out == 0.0 and pulse is None)
    need_totals = return_totals or (run_sanity_checks and diffusion_only)
    if need_totals:
        B_tot = np.empty(nt + 1, dtype=float)
        B_tot[0] = _domain_integral(U, dx=dx, dy=dy)

    for n in range(1, nt + 1):
        t_now = time[n]
        h_y = fishing_profile_y(y, t_now, y_boundary, h_in, h_out, pulse=pulse)
        H = h_y[:, None]

        lap_u = laplacian_2d_periodicx_neumanny(U, dx=dx, dy=dy)
        U = U + dt * (r * U * (1.0 - U / K) - H * U + D * lap_u)

        cur_min = float(U.min())
        if cur_min < min_u:
            min_u = cur_min
        U = np.maximum(U, 0.0)

        if need_totals:
            B_tot[n] = _domain_integral(U, dx=dx, dy=dy)

        if n in snap_steps:
            for t_req in snap_steps[n]:
                snapshots[t_req] = U.copy()

    if run_sanity_checks:
        _check_uniform_single(r=r, K=K, D=D, dt=dt, dx=dx, dy=dy, nx=nx, ny=ny, uniform_tol=uniform_tol)
        if diffusion_only and need_totals:
            _check_mass_drift(
                B_tot[0],
                B_tot[-1],
                tol=mass_tol,
                label="single-species diffusion-only mass drift",
            )

    result: dict[str, Any] = {
        "x": x,
        "y": y,
        "time": time,
        "times": time,
        "snapshots": snapshots,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "min_u": min_u,
        "y_boundary": y_boundary,
        "snapshot_times": np.array(snap_targets, dtype=float),
    }
    if return_totals:
        result["B_tot"] = B_tot if need_totals else None
    return result


def simulate_competing_2d(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    D1: float,
    D2: float,
    r1: float,
    r2: float,
    K1: float,
    K2: float,
    a12: float,
    a21: float,
    T_end: float,
    y_boundary: float = 200.0,
    h1_in: float = 0.0,
    h1_out: float = 0.0,
    h2_in: float = 0.0,
    h2_out: float = 0.0,
    pulse1: dict | None = None,
    pulse2: dict | None = None,
    snapshot_times: list[float] | None = None,
    noise_eps: float = 0.02,
    seed: int | None = None,
    return_totals: bool = True,
    run_sanity_checks: bool = True,
    mass_tol: float = 1e-3,
    uniform_tol: float = 1e-10,
) -> dict[str, Any]:
    """Simulate 2D Lotka-Volterra competition + diffusion + y-dependent fishing."""
    _validate_common_inputs(Lx=Lx, Ly=Ly, nx=nx, ny=ny, T_end=T_end)
    if D1 <= 0.0 or D2 <= 0.0:
        raise ValueError("D1 and D2 must be positive for explicit diffusion timestep rule.")
    if K1 <= 0.0 or K2 <= 0.0:
        raise ValueError("K1 and K2 must be positive.")
    if r1 < 0.0 or r2 < 0.0:
        raise ValueError("r1 and r2 must be nonnegative.")

    x, y, dx, dy = _build_grid(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    dt_diff = 0.2 * min(dx * dx, dy * dy) / (4.0 * max(D1, D2))
    dt_reac = 0.2 / max(r1, r2, 1e-12)
    dt_raw = min(dt_diff, dt_reac)
    nt = int(np.ceil(T_end / dt_raw))
    dt = T_end / nt
    time = np.linspace(0.0, T_end, nt + 1)

    snap_targets, snap_steps = _snapshot_schedule(snapshot_times=snapshot_times, T_end=T_end, dt=dt, nt=nt)

    rng = np.random.default_rng(seed)
    U, V = _init_competing_fields(
        x=x,
        y=y,
        K1=K1,
        K2=K2,
        noise_eps=noise_eps,
        rng=rng,
    )
    min_u = float(U.min())
    min_v = float(V.min())

    snapshots_u: dict[float, np.ndarray] = {}
    snapshots_v: dict[float, np.ndarray] = {}
    if 0 in snap_steps:
        for t_req in snap_steps[0]:
            snapshots_u[t_req] = U.copy()
            snapshots_v[t_req] = V.copy()

    diffusion_only = (
        r1 == 0.0
        and r2 == 0.0
        and h1_in == 0.0
        and h1_out == 0.0
        and h2_in == 0.0
        and h2_out == 0.0
        and pulse1 is None
        and pulse2 is None
    )
    need_totals = return_totals or (run_sanity_checks and diffusion_only)
    if need_totals:
        B1_tot = np.empty(nt + 1, dtype=float)
        B2_tot = np.empty(nt + 1, dtype=float)
        B1_tot[0] = _domain_integral(U, dx=dx, dy=dy)
        B2_tot[0] = _domain_integral(V, dx=dx, dy=dy)

    for n in range(1, nt + 1):
        t_now = time[n]

        h1_y = fishing_profile_y(y, t_now, y_boundary, h1_in, h1_out, pulse=pulse1)
        h2_y = fishing_profile_y(y, t_now, y_boundary, h2_in, h2_out, pulse=pulse2)
        H1 = h1_y[:, None]
        H2 = h2_y[:, None]

        lap_u = laplacian_2d_periodicx_neumanny(U, dx=dx, dy=dy)
        lap_v = laplacian_2d_periodicx_neumanny(V, dx=dx, dy=dy)

        dU = r1 * U * (1.0 - (U + a12 * V) / K1) - H1 * U + D1 * lap_u
        dV = r2 * V * (1.0 - (V + a21 * U) / K2) - H2 * V + D2 * lap_v
        U = U + dt * dU
        V = V + dt * dV

        cur_min_u = float(U.min())
        cur_min_v = float(V.min())
        if cur_min_u < min_u:
            min_u = cur_min_u
        if cur_min_v < min_v:
            min_v = cur_min_v

        U = np.maximum(U, 0.0)
        V = np.maximum(V, 0.0)

        if need_totals:
            B1_tot[n] = _domain_integral(U, dx=dx, dy=dy)
            B2_tot[n] = _domain_integral(V, dx=dx, dy=dy)

        if n in snap_steps:
            for t_req in snap_steps[n]:
                snapshots_u[t_req] = U.copy()
                snapshots_v[t_req] = V.copy()

    if run_sanity_checks:
        _check_uniform_competing(
            r1=r1,
            r2=r2,
            K1=K1,
            K2=K2,
            a12=a12,
            a21=a21,
            D1=D1,
            D2=D2,
            dt=dt,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            uniform_tol=uniform_tol,
        )
        if diffusion_only and need_totals:
            _check_mass_drift(
                B1_tot[0],
                B1_tot[-1],
                tol=mass_tol,
                label="species U diffusion-only mass drift",
            )
            _check_mass_drift(
                B2_tot[0],
                B2_tot[-1],
                tol=mass_tol,
                label="species V diffusion-only mass drift",
            )

    result: dict[str, Any] = {
        "x": x,
        "y": y,
        "time": time,
        "times": time,
        "snapshots_u": snapshots_u,
        "snapshots_v": snapshots_v,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "min_u": min_u,
        "min_v": min_v,
        "y_boundary": y_boundary,
        "snapshot_times": np.array(snap_targets, dtype=float),
    }
    if return_totals:
        result["B1_tot"] = B1_tot if need_totals else None
        result["B2_tot"] = B2_tot if need_totals else None
    return result


def _build_grid(Lx: float, Ly: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    return x, y, dx, dy


def _validate_common_inputs(Lx: float, Ly: float, nx: int, ny: int, T_end: float) -> None:
    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError("Lx and Ly must be positive.")
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must both be >= 3.")
    if T_end <= 0.0:
        raise ValueError("T_end must be positive.")


def _snapshot_schedule(
    snapshot_times: list[float] | None,
    T_end: float,
    dt: float,
    nt: int,
) -> tuple[list[float], dict[int, list[float]]]:
    if snapshot_times is None:
        snapshot_times = [0.0, 0.25 * T_end, 0.5 * T_end, 0.75 * T_end, T_end]

    clean_times: list[float] = []
    for t in snapshot_times:
        t_float = float(t)
        if t_float < 0.0 or t_float > T_end:
            raise ValueError(f"snapshot time {t_float} outside [0, T_end].")
        clean_times.append(t_float)
    clean_times = sorted(set(clean_times))

    by_step: dict[int, list[float]] = {}
    for t_req in clean_times:
        step = int(np.clip(np.rint(t_req / dt), 0, nt))
        by_step.setdefault(step, []).append(t_req)
    return clean_times, by_step


def _domain_integral(U: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(U) * dx * dy)


def _init_single_field(
    x: np.ndarray,
    y: np.ndarray,
    K: float,
    ic_mode: str,
    noise_eps: float,
    n_blobs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    X, Y = np.meshgrid(x, y)
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])

    if ic_mode == "noisy_baseline":
        U = 0.9 * K + noise_eps * K * (rng.random((len(y), len(x))) - 0.5)
    elif ic_mode == "blobs":
        U = 0.15 * K + noise_eps * K * (rng.random((len(y), len(x))) - 0.5)
        for _ in range(n_blobs):
            x0 = rng.uniform(0.0, Lx)
            y0 = rng.uniform(0.0, Ly)
            sx = rng.uniform(0.04 * Lx, 0.14 * Lx)
            sy = rng.uniform(0.04 * Ly, 0.14 * Ly)
            amp = rng.uniform(0.08 * K, 0.30 * K)
            U += amp * np.exp(-((X - x0) / sx) ** 2 - ((Y - y0) / sy) ** 2)
    else:
        raise ValueError("ic_mode must be 'noisy_baseline' or 'blobs'.")

    # Local perturbation patch to trigger 2D fronts and avoid x-invariant evolution.
    x_patch = rng.uniform(0.1 * Lx, 0.9 * Lx)
    y_patch = rng.uniform(0.1 * Ly, 0.9 * Ly)
    sx_patch = max(0.03 * Lx, 1e-12)
    sy_patch = max(0.03 * Ly, 1e-12)
    amp_patch = (0.12 + 0.08 * rng.random()) * K
    U += amp_patch * np.exp(-((X - x_patch) / sx_patch) ** 2 - ((Y - y_patch) / sy_patch) ** 2)

    return np.maximum(U, 0.0)


def _init_competing_fields(
    x: np.ndarray,
    y: np.ndarray,
    K1: float,
    K2: float,
    noise_eps: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, y)
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])

    # Species U: coastal 2D blobs near y=0 with x variation.
    y_u = max(0.04 * Ly, 1e-9)
    x_u1 = rng.uniform(0.15 * Lx, 0.85 * Lx)
    x_u2 = rng.uniform(0.0, Lx)
    U = 0.70 * K1 * np.exp(-((X - x_u1) / (0.12 * Lx + 1e-12)) ** 2 - ((Y - y_u) / (0.10 * Ly + 1e-12)) ** 2)
    U += 0.45 * K1 * np.exp(
        -((X - x_u2) / (0.08 * Lx + 1e-12)) ** 2
        - ((Y - (y_u + 0.08 * Ly)) / (0.07 * Ly + 1e-12)) ** 2
    )

    # Species V: offshore 2D blobs, target y in [250, 350] when domain allows.
    if Ly >= 350.0:
        y_v = rng.uniform(250.0, 350.0)
    else:
        y_v = 0.58 * Ly
    x_v1 = rng.uniform(0.1 * Lx, 0.9 * Lx)
    x_v2 = rng.uniform(0.0, Lx)
    V = 0.65 * K2 * np.exp(-((X - x_v1) / (0.11 * Lx + 1e-12)) ** 2 - ((Y - y_v) / (0.11 * Ly + 1e-12)) ** 2)
    V += 0.35 * K2 * np.exp(
        -((X - x_v2) / (0.09 * Lx + 1e-12)) ** 2
        - ((Y - (y_v - 0.10 * Ly)) / (0.08 * Ly + 1e-12)) ** 2
    )

    U += noise_eps * K1 * (rng.random(U.shape) - 0.5)
    V += noise_eps * K2 * (rng.random(V.shape) - 0.5)

    return np.maximum(U, 0.0), np.maximum(V, 0.0)


def _check_mass_drift(m0: float, m1: float, tol: float, label: str) -> None:
    denom = max(abs(m0), 1e-12)
    rel_drift = abs(m1 - m0) / denom
    if rel_drift > tol:
        raise RuntimeError(f"{label} too large: relative drift={rel_drift:.3e} > tol={tol:.3e}")


def _check_uniform_single(
    r: float,
    K: float,
    D: float,
    dt: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    uniform_tol: float,
) -> None:
    U0 = np.full((ny, nx), 0.37 * K, dtype=float)
    lap = laplacian_2d_periodicx_neumanny(U0, dx=dx, dy=dy)
    if float(np.max(np.abs(lap))) > uniform_tol:
        raise RuntimeError("Uniform single-species Laplacian check failed.")

    pde_next = U0 + dt * (r * U0 * (1.0 - U0 / K) + D * lap)
    ode_next = U0 + dt * (r * U0 * (1.0 - U0 / K))
    if float(np.max(np.abs(pde_next - ode_next))) > 10.0 * uniform_tol:
        raise RuntimeError("Uniform single-species ODE reduction check failed.")


def _check_uniform_competing(
    r1: float,
    r2: float,
    K1: float,
    K2: float,
    a12: float,
    a21: float,
    D1: float,
    D2: float,
    dt: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    uniform_tol: float,
) -> None:
    U0 = np.full((ny, nx), 0.35 * K1, dtype=float)
    V0 = np.full((ny, nx), 0.32 * K2, dtype=float)

    lap_u = laplacian_2d_periodicx_neumanny(U0, dx=dx, dy=dy)
    lap_v = laplacian_2d_periodicx_neumanny(V0, dx=dx, dy=dy)
    if float(np.max(np.abs(lap_u))) > uniform_tol or float(np.max(np.abs(lap_v))) > uniform_tol:
        raise RuntimeError("Uniform two-species Laplacian check failed.")

    pde_u_next = U0 + dt * (r1 * U0 * (1.0 - (U0 + a12 * V0) / K1) + D1 * lap_u)
    pde_v_next = V0 + dt * (r2 * V0 * (1.0 - (V0 + a21 * U0) / K2) + D2 * lap_v)

    ode_u_next = U0 + dt * (r1 * U0 * (1.0 - (U0 + a12 * V0) / K1))
    ode_v_next = V0 + dt * (r2 * V0 * (1.0 - (V0 + a21 * U0) / K2))

    if float(np.max(np.abs(pde_u_next - ode_u_next))) > 10.0 * uniform_tol:
        raise RuntimeError("Uniform two-species ODE reduction check failed for U.")
    if float(np.max(np.abs(pde_v_next - ode_v_next))) > 10.0 * uniform_tol:
        raise RuntimeError("Uniform two-species ODE reduction check failed for V.")


__all__ = [
    "laplacian_2d_periodicx_neumanny",
    "fishing_profile_y",
    "simulate_single_2d",
    "simulate_competing_2d",
]
