"""Spatial discretisation and PDE solvers for fish population dynamics."""

from __future__ import annotations

import numpy as np


def laplacian_neumann(u: np.ndarray, ds: float) -> np.ndarray:
    """Second spatial derivative with no-flux (Neumann) boundaries.

    Ghost-point equivalence gives:
        left  boundary: d²u/ds² ≈ 2*(u[1]  - u[0] ) / ds²
        right boundary: d²u/ds² ≈ 2*(u[-2] - u[-1]) / ds²
    """
    d2u = np.empty_like(u)
    inv_ds2 = 1.0 / ds**2
    # interior points
    d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) * inv_ds2
    # boundaries (Neumann via ghost point)
    d2u[0] = 2.0 * (u[1] - u[0]) * inv_ds2
    d2u[-1] = 2.0 * (u[-2] - u[-1]) * inv_ds2
    return d2u


def simulate_rd_fishing(
    L: float,
    N: int,
    D: float,
    r: float,
    K: float,
    T_end: float,
    s_boundary: float = 200.0,
    h_in: float = 0.0,
    h_out: float = 0.2,
    pulse: dict | None = None,
    u0_type: str = "gaussian",
    u0_uniform: float | None = None,
    gaussian_center: float = 150.0,
    gaussian_scale: float = 50.0,
    snapshot_times: list[float] | None = None,
    dt_safety: float = 0.45,
    store_full: bool = False,
    full_n_frames: int = 300,
    h_in_schedule: np.ndarray | None = None,
    h_out_schedule: np.ndarray | None = None,
) -> dict:
    """Simulate reaction-diffusion PDE with spatially varying fishing.

    Parameters
    ----------
    L, N          : domain length and grid points
    D, r, K       : diffusion, growth rate, carrying capacity
    T_end         : simulation end time
    s_boundary    : policy boundary (inshore/offshore split)
    h_in, h_out   : fishing rates for s <= s_boundary and s > s_boundary
                    (used as constants when schedules are not provided)
    pulse         : optional dict with keys "t_start", "t_end", "h_out_pulse"
    u0_type       : "gaussian" or "uniform"
    u0_uniform    : initial value if u0_type == "uniform" (default 0.2*K)
    gaussian_center: centre of Gaussian IC (offshore distance of peak density)
    gaussian_scale: width parameter for Gaussian IC
    snapshot_times: list of times at which to store spatial snapshots
    dt_safety     : safety factor for diffusion stability
    h_in_schedule : shape (n_years,) array of annual inshore harvest rates;
                    overrides h_in when provided (year = floor(t))
    h_out_schedule: shape (n_years,) array of annual offshore harvest rates;
                    overrides h_out when provided (year = floor(t))

    Returns
    -------
    dict with keys: s, time, snapshots, B_in, B_out, B_tot, Catch,
                    CumCatch, ds, dt, nt, min_u, s_boundary,
                    h_in_schedule, h_out_schedule
                    (+ u_full, t_full when store_full=True)
    """
    if snapshot_times is None:
        snapshot_times = [0.0, 10.0, 20.0, 40.0, 60.0]

    ds = L / (N - 1)
    s = np.linspace(0.0, L, N)

    # -- dt: diffusion stability AND reaction/fishing safety -------------
    dt = dt_safety * ds**2 / (2.0 * D)
    h_pulse_val = pulse["h_out_pulse"] if pulse else 0.0
    h_in_max  = float(np.max(h_in_schedule))  if h_in_schedule  is not None else h_in
    h_out_max = float(np.max(h_out_schedule)) if h_out_schedule is not None else h_out
    max_rate = max(r, h_in_max, h_out_max, h_pulse_val, 1e-12)
    dt = min(dt, 0.2 / max_rate)
    nt = int(np.ceil(T_end / dt))
    dt = T_end / nt

    # -- boundary index --------------------------------------------------
    i_bnd = int(np.searchsorted(s, s_boundary, side="right")) - 1

    # -- initial condition ------------------------------------------------
    if u0_type == "gaussian":
        u = np.exp(-((s - gaussian_center) / gaussian_scale) ** 2)
    else:
        u = (u0_uniform if u0_uniform is not None else 0.2 * K) * np.ones(N)

    # -- base fishing-rate array -----------------------------------------
    h_base = np.empty(N)
    h_base[: i_bnd + 1] = h_in
    h_base[i_bnd + 1 :] = h_out

    # -- full space-time storage (subsampled) ----------------------------
    if store_full:
        save_every = max(1, nt // full_n_frames)
        n_saved = nt // save_every + 1
        u_full = np.empty((n_saved, N))
        t_full = np.empty(n_saved)
        u_full[0] = u.copy()
        t_full[0] = 0.0
        full_idx = 1

    # -- storage ---------------------------------------------------------
    snap_set = set(snapshot_times)
    snapshots: dict[float, np.ndarray] = {}
    if 0.0 in snap_set:
        snapshots[0.0] = u.copy()

    time_arr = np.empty(nt + 1)
    B_in = np.empty(nt + 1)
    B_out = np.empty(nt + 1)
    B_tot = np.empty(nt + 1)
    Catch = np.empty(nt + 1)
    CumCatch = np.empty(nt + 1)

    time_arr[0] = 0.0
    B_in[0] = np.trapezoid(u[: i_bnd + 1], s[: i_bnd + 1])
    B_out[0] = np.trapezoid(u[i_bnd:], s[i_bnd:])
    B_tot[0] = np.trapezoid(u, s)
    Catch[0] = np.trapezoid(h_base * u, s)
    CumCatch[0] = 0.0

    min_u = float(u.min())

    # -- time stepping ---------------------------------------------------
    for n in range(1, nt + 1):
        t_now = n * dt

        # fishing rate at this time
        if h_in_schedule is not None:
            # annual stochastic schedule: look up current year
            year = min(int(t_now), len(h_in_schedule) - 1)
            h_arr = np.empty(N)
            h_arr[: i_bnd + 1] = h_in_schedule[year]
            h_arr[i_bnd + 1 :] = h_out_schedule[year]
        elif pulse and pulse["t_start"] <= t_now <= pulse["t_end"]:
            h_arr = h_base.copy()
            h_arr[i_bnd + 1 :] = pulse["h_out_pulse"]
        else:
            h_arr = h_base

        lap = laplacian_neumann(u, ds)
        u = u + dt * (r * u * (1.0 - u / K) - h_arr * u + D * lap)

        # diagnostics
        cur_min = float(u.min())
        if cur_min < min_u:
            min_u = cur_min

        time_arr[n] = t_now
        B_in[n] = np.trapezoid(u[: i_bnd + 1], s[: i_bnd + 1])
        B_out[n] = np.trapezoid(u[i_bnd:], s[i_bnd:])
        B_tot[n] = np.trapezoid(u, s)
        Catch[n] = np.trapezoid(h_arr * u, s)
        CumCatch[n] = CumCatch[n - 1] + dt * Catch[n - 1]

        # store full space-time data (subsampled)
        if store_full and n % save_every == 0 and full_idx < n_saved:
            u_full[full_idx] = u.copy()
            t_full[full_idx] = t_now
            full_idx += 1

        # capture snapshots
        for ts in list(snap_set):
            if abs(t_now - ts) < 0.5 * dt:
                snapshots[ts] = u.copy()
                snap_set.discard(ts)

    if min_u < -1e-10:
        print(f"  WARNING: min(u) = {min_u:.3e} (negative!)")

    result = {
        "s": s,
        "time": time_arr,
        "snapshots": snapshots,
        "B_in": B_in,
        "B_out": B_out,
        "B_tot": B_tot,
        "Catch": Catch,
        "CumCatch": CumCatch,
        "ds": ds,
        "dt": dt,
        "nt": nt,
        "min_u": min_u,
        "s_boundary": s_boundary,
        "h_in_schedule": h_in_schedule,
        "h_out_schedule": h_out_schedule,
    }

    if store_full:
        result["u_full"] = u_full[:full_idx]
        result["t_full"] = t_full[:full_idx]

    return result
