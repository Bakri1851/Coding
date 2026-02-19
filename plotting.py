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
        ax.plot(t_eval, x_num, linewidth=2.0, label=f"RK45, x\u0303\u2080={x0:.3f}")
        ax.plot(t_eval, x_an, "--", linewidth=1.2, label=f"Analytic, x\u0303\u2080={x0:.3f}")

    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="x\u0303*=0")
    ax.axhline(K, color="gray", linestyle="-.", linewidth=1.2, label="x\u0303*=1")

    ax.set_title("Logistic Growth (ND): Multiple Initial Conditions x\u0303\u2080")
    ax.set_xlabel("Dimensionless time \u03c4")
    ax.set_ylabel("Dimensionless population x\u0303(\u03c4)")
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

    ax.axhline(K, color="gray", linestyle="-.", linewidth=1.2, label="x\u0303*=1")
    ax.set_title(f"Logistic Growth (ND): r-sweep (x\u0303\u2080={x0:.3f})")
    ax.set_xlabel("Dimensionless time \u03c4")
    ax.set_ylabel("Dimensionless population x\u0303(\u03c4)")
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
        u_snap = res["snapshots"][t_snap]
        line, = ax.plot(res["s"], u_snap, linewidth=2.0, label=f"\u03c4 = {t_snap:g}")
        # Mark the peak density location
        peak_idx = np.argmax(u_snap)
        s_peak = res["s"][peak_idx]
        u_peak = u_snap[peak_idx]
        ax.plot(s_peak, u_peak, "o", color=line.get_color(), markersize=8,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate(f"\u03be={s_peak:.3f}", (s_peak, u_peak),
                    textcoords="offset points", xytext=(8, 6), fontsize=7.5,
                    color=line.get_color(), fontweight="bold")
    if K is not None:
        ax.axhline(K, color="gray", linestyle="--", linewidth=1.2, label=f"x* = {K:g}")
    ax.axvline(res["s_boundary"], color="red", linestyle=":", linewidth=1.2,
               label=f"\u03be_bnd = {res['s_boundary']:.3f}")
    ax.set_title(f"Density Snapshots — {title}")
    ax.set_xlabel("Dimensionless position \u03be")
    ax.set_ylabel("Dimensionless density x(\u03be, \u03c4)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_biomass(res: dict, title: str) -> plt.Figure:
    """Biomass time-series plot for one scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res["time"], res["B_in"], linewidth=2.0, label="B_in (\u03be \u2264 \u03be_bnd)")
    ax.plot(res["time"], res["B_out"], linewidth=2.0, label="B_out (\u03be > \u03be_bnd)")
    ax.plot(res["time"], res["B_tot"], linewidth=2.0, linestyle="--", label="B_tot (total)")
    ax.set_title(f"Biomass Time Series — {title}")
    ax.set_xlabel("Dimensionless time \u03c4")
    ax.set_ylabel("Dimensionless biomass \u222bx d\u03be")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_catch(res: dict, title: str) -> plt.Figure:
    """Catch rate + cumulative catch (twin axes) for one scenario."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_catch = "tab:blue"
    ax1.plot(res["time"], res["Catch"], linewidth=2.0, color=color_catch, label="Catch rate")
    ax1.set_xlabel("Dimensionless time \u03c4")
    ax1.set_ylabel("Dimensionless catch rate", color=color_catch)
    ax1.tick_params(axis="y", labelcolor=color_catch)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_cum = "tab:orange"
    ax2.plot(res["time"], res["CumCatch"], linewidth=2.0, color=color_cum, label="Cumulative catch")
    ax2.set_ylabel("Cumulative dimensionless catch", color=color_cum)
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
    # Trace peak density ridge along the surface
    peak_idx = np.argmax(u_plot, axis=1)
    peak_s = s_plot[peak_idx]
    peak_u = u_plot[np.arange(len(t_plot)), peak_idx]
    ax3d.plot(peak_s, t_plot, peak_u, color="red", linewidth=2.5, label="Peak density core")
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.set_xlabel("Dimensionless position \u03be")
    ax3d.set_ylabel("Dimensionless time \u03c4")
    ax3d.set_zlabel("Dimensionless density x")
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

    # Compute peak ridge trace
    peak_idx = np.argmax(u_plot, axis=1)
    peak_s = s_plot[peak_idx]
    peak_u = u_plot[np.arange(len(t_plot)), peak_idx]

    fig = go.Figure(
        data=[
            go.Surface(
                x=S_mesh,
                y=T_mesh,
                z=u_plot,
                colorscale="Viridis",
                colorbar=dict(title="Dimensionless density x(\u03be, \u03c4)"),
            ),
            go.Scatter3d(
                x=peak_s, y=t_plot, z=peak_u,
                mode="lines",
                line=dict(color="red", width=6),
                name="Peak density core",
            ),
        ]
    )
    fig.update_layout(
        title=f"3D Density Surface (Interactive) — {title}",
        scene=dict(
            xaxis_title="Dimensionless position \u03be",
            yaxis_title="Dimensionless time \u03c4",
            zaxis_title="Dimensionless density x",
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
            label=f"\u03be_boundary = {s_boundary:.3f}",
        )
    # Trace peak density location over time
    peak_s = s_arr[np.argmax(u_full, axis=1)]
    ax.plot(peak_s, t_arr, color="magenta", linewidth=2.0, linestyle="--",
            label="Peak density core")
    ax.set_xlabel("Dimensionless position \u03be")
    ax.set_ylabel("Dimensionless time \u03c4")
    ax.set_title(f"Density Heatmap — {title}")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Dimensionless density x(\u03be, \u03c4)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig
