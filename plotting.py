"""Reusable plotting functions for fish population model visualisations."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ode_models import logistic_numeric

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# ODE plots
# ---------------------------------------------------------------------------

def plot_initial_condition_comparison(
    t_eval: np.ndarray,
    r: float,
    K: float,
    x0_list: list[float],
) -> plt.Figure:
    """Plot x(t) for multiple initial conditions (RK45 vs analytic)."""
    from ode_models import logistic_analytic

    fig, ax = plt.subplots(figsize=(10, 6))

    for x0 in x0_list:
        x_num = logistic_numeric(t_eval=t_eval, r=r, K=K, x0=x0)
        x_an = logistic_analytic(t=t_eval, r=r, K=K, x0=x0)
        ax.plot(t_eval, x_num, linewidth=2.0, label=f"RK45, x0={x0:.1f}")
        ax.plot(t_eval, x_an, "--", linewidth=1.2, label=f"Analytic, x0={x0:.1f}")

    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="Equilibrium x=0")
    ax.axhline(K, color="gray", linestyle="-.", linewidth=1.2, label=f"Equilibrium x=K={K:g}")

    ax.set_title("Logistic Growth: Multiple Initial Conditions")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Population x(t)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    return fig


def plot_r_sweep(
    t_eval: np.ndarray,
    r_values: list[float],
    K: float,
    x0: float,
) -> plt.Figure:
    """Compare growth speed for different intrinsic rates r."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in r_values:
        x_num = logistic_numeric(t_eval=t_eval, r=r, K=K, x0=x0)
        ax.plot(t_eval, x_num, linewidth=2.0, label=f"r={r:.2f}")

    ax.axhline(K, color="gray", linestyle="-.", linewidth=1.2, label=f"K={K:g}")
    ax.set_title(f"Logistic Growth Speed for Different r (x0={x0:.1f})")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Population x(t)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fishing-scenario plots
# ---------------------------------------------------------------------------

def plot_snapshots(res: dict, title: str, K: float | None = None) -> plt.Figure:
    """Density snapshot plot for one scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_snap in sorted(res["snapshots"]):
        ax.plot(res["s"], res["snapshots"][t_snap],
                linewidth=2.0, label=f"t = {t_snap:g}")
    if K is not None:
        ax.axhline(K, color="gray", linestyle="--", linewidth=1.2, label=f"K = {K:g}")
    ax.axvline(res["s_boundary"], color="red", linestyle=":", linewidth=1.2,
               label=f"boundary = {res['s_boundary']:g}")
    ax.set_title(f"Density Snapshots — {title}")
    ax.set_xlabel("Offshore distance s")
    ax.set_ylabel("Population density u(s, t)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_biomass(res: dict, title: str) -> plt.Figure:
    """Biomass time-series plot for one scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res["time"], res["B_in"], linewidth=2.0, label="B_in (inshore)")
    ax.plot(res["time"], res["B_out"], linewidth=2.0, label="B_out (offshore)")
    ax.plot(res["time"], res["B_tot"], linewidth=2.0, linestyle="--", label="B_tot (total)")
    ax.set_title(f"Biomass Time Series — {title}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Biomass (integrated density)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_catch(res: dict, title: str) -> plt.Figure:
    """Catch rate + cumulative catch (twin axes) for one scenario."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_catch = "tab:blue"
    ax1.plot(res["time"], res["Catch"], linewidth=2.0, color=color_catch, label="Catch rate")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Catch rate (density/time)", color=color_catch)
    ax1.tick_params(axis="y", labelcolor=color_catch)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_cum = "tab:orange"
    ax2.plot(res["time"], res["CumCatch"], linewidth=2.0, color=color_cum, label="Cumulative catch")
    ax2.set_ylabel("Cumulative catch", color=color_cum)
    ax2.tick_params(axis="y", labelcolor=color_cum)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(f"Catch — {title}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D surface and heatmap plots
# ---------------------------------------------------------------------------

def downsample_surface_data(
    s_arr: np.ndarray,
    t_arr: np.ndarray,
    u_full: np.ndarray,
    max_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce resolution for 3D surface plotting."""
    if u_full.shape != (len(t_arr), len(s_arr)):
        raise ValueError(
            f"u_full shape {u_full.shape} does not match (len(t_arr), len(s_arr)) "
            f"= ({len(t_arr)}, {len(s_arr)})."
        )

    if len(s_arr) <= max_points and len(t_arr) <= max_points:
        return s_arr, t_arr, u_full

    s_idx = np.linspace(0, len(s_arr) - 1, num=min(len(s_arr), max_points), dtype=int)
    t_idx = np.linspace(0, len(t_arr) - 1, num=min(len(t_arr), max_points), dtype=int)
    return s_arr[s_idx], t_arr[t_idx], u_full[np.ix_(t_idx, s_idx)]


def plot_3d_surface(
    s_arr: np.ndarray,
    t_arr: np.ndarray,
    u_full: np.ndarray,
    title: str,
    s_boundary: float | None = None,
    max_points: int = 200,
) -> plt.Figure:
    """Matplotlib 3D surface plot of the density field."""
    s_plot, t_plot, u_plot = downsample_surface_data(s_arr, t_arr, u_full, max_points)
    S_mesh, T_mesh = np.meshgrid(s_plot, t_plot)

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot_surface(S_mesh, T_mesh, u_plot, cmap="viridis", edgecolor="none", alpha=0.9)
    ax3d.set_xlabel("Offshore distance s")
    ax3d.set_ylabel("Time t")
    ax3d.set_zlabel("Fish density u")
    ax3d.set_title(f"3D Density Surface — {title}")
    ax3d.view_init(elev=25, azim=-50)
    fig.tight_layout()
    return fig


def plot_3d_surface_plotly(
    s_arr: np.ndarray,
    t_arr: np.ndarray,
    u_full: np.ndarray,
    title: str,
    max_points: int = 200,
):
    """Interactive Plotly 3D surface (skips silently if Plotly unavailable)."""
    if not HAS_PLOTLY:
        return None

    s_plot, t_plot, u_plot = downsample_surface_data(s_arr, t_arr, u_full, max_points)
    S_mesh, T_mesh = np.meshgrid(s_plot, t_plot)

    fig = go.Figure(
        data=[
            go.Surface(
                x=S_mesh,
                y=T_mesh,
                z=u_plot,
                colorscale="Viridis",
                colorbar=dict(title="Fish density u(s, t)"),
            )
        ]
    )
    fig.update_layout(
        title=f"3D Density Surface (Interactive) — {title}",
        scene=dict(
            xaxis_title="Offshore distance s",
            yaxis_title="Time t",
            zaxis_title="Fish density u",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=700,
    )
    fig.update_scenes(camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)))
    fig.show()
    return fig


def plot_heatmap(
    s_arr: np.ndarray,
    t_arr: np.ndarray,
    u_full: np.ndarray,
    title: str,
    s_boundary: float | None = None,
) -> plt.Figure:
    """2D heatmap of the density field."""
    fig, ax = plt.subplots(figsize=(12, 6))
    pcm = ax.pcolormesh(s_arr, t_arr, u_full, cmap="viridis", shading="auto")
    if s_boundary is not None:
        ax.axvline(
            s_boundary, color="red", linestyle="--", linewidth=1.5,
            label=f"Policy boundary (s = {s_boundary:g})",
        )
    ax.set_xlabel("Offshore distance s")
    ax.set_ylabel("Time t")
    ax.set_title(f"Density Heatmap — {title}")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Fish density u(s, t)")
    if s_boundary is not None:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig
