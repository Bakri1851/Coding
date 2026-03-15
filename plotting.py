"""Reusable plotting functions for fish population model visualisations."""

from __future__ import annotations

import re
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

def _extract_boundaries(res: dict) -> list[tuple[str, float, str, str]]:
    """Return available policy boundaries in a plotting-friendly format."""
    if "s_boundary" in res:
        return [("EEZ (200-mile limit)", float(res["s_boundary"]), "red", ":")]

    boundaries: list[tuple[str, float, str, str]] = []
    if "s_boundary1" in res:
        boundaries.append(("\u03be_bnd1", float(res["s_boundary1"]), "#1f77b4", ":"))
    if "s_boundary2" in res:
        boundaries.append(("\u03be_bnd2", float(res["s_boundary2"]), "red", "--"))
    return boundaries


def _add_harvest_start_marker(ax: plt.Axes, res: dict) -> None:
    """Draw a vertical marker at harvest onset when present."""
    harvest_start_time = float(res.get("harvest_start_time", 0.0))
    has_catch = False
    if "Catch" in res:
        has_catch = bool(np.any(np.asarray(res["Catch"]) > 0.0))
    elif "Catch1" in res or "Catch2" in res:
        catch1 = np.asarray(res.get("Catch1", 0.0))
        catch2 = np.asarray(res.get("Catch2", 0.0))
        has_catch = bool(np.any(catch1 > 0.0) or np.any(catch2 > 0.0))
    if harvest_start_time > 0.0 and has_catch:
        ax.axvline(
            harvest_start_time,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Harvest starts (\u03c4={harvest_start_time:.2f})",
        )


def _two_species_palette() -> dict[str, tuple[float, float, float] | str]:
    """Cold tones for species 1 and warm tones for species 2."""
    return {
        "sp1_tot": "#1D4E89",  # deep ocean blue
        "sp1_in": "#2A9D8F",   # teal
        "sp1_out": "#56B4E9",  # sky blue
        "sp2_tot": "#C75B12",  # burnt orange
        "sp2_in": "#E69F00",   # amber
        "sp2_out": "#E76F51",  # coral
    }


def plot_snapshots(res: dict, title: str, K: float | None = None) -> plt.Figure:
    """Density snapshot plot for one scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_snap in sorted(res["snapshots"]):
        u_snap = res["snapshots"][t_snap]
        ax.plot(res["s"], u_snap, linewidth=2.0, label=f"\u03c4 = {t_snap:g}")
    if K is not None:
        ax.axhline(K, color="gray", linestyle="--", linewidth=1.2, label=f"x* = {K:g}")
    for bnd_label, bnd_pos, bnd_color, bnd_ls in _extract_boundaries(res):
        ax.axvline(
            bnd_pos,
            color=bnd_color,
            linestyle=bnd_ls,
            linewidth=1.2,
            label=f"{bnd_label} = {bnd_pos:.3f}",
        )
    #ax.set_title(f"Density Snapshots — {title}")
    ax.set_xlabel("Dimensionless position \u03be")
    ax.set_ylabel("Dimensionless density x(\u03be, \u03c4)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_biomass(res: dict, title: str) -> plt.Figure:
    """Biomass time-series plot for one scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res["time"], res["B_in"], linewidth=2.0, label="International (\u03be \u2264 \u03be_E)")
    ax.plot(res["time"], res["B_out"], linewidth=2.0, label="EEZ (\u03be > \u03be_E)")
    ax.plot(res["time"], res["B_tot"], linewidth=2.0, linestyle="--", label="B_tot (total)")
    _add_harvest_start_marker(ax, res)
    #ax.set_title(f"Biomass Time Series — {title}")
    ax.set_xlabel("Dimensionless time \u03c4")
    ax.set_ylabel("  x")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_catch(res: dict, title: str) -> plt.Figure:
    """Catch rate + cumulative catch (twin axes) for one scenario."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_catch = "tab:blue"
    ax1.plot(res["time"], res["Catch"], linewidth=2.0, color=color_catch, label="Catch rate")
    _add_harvest_start_marker(ax1, res)
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
            label="EEZ (200-mile limit)",
        )
    ax.set_xlabel("Dimensionless position \u03be")
    ax.set_ylabel("Dimensionless time \u03c4")
    #ax.set_title(f"Density Heatmap — {title}")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Dimensionless density x(\u03be, \u03c4)")
    if s_boundary is not None:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Two-species competing plots (Section 4)
# ---------------------------------------------------------------------------

def plot_snapshots_2s(
    res: dict,
    title: str,
    K1: float | None = None,
    K2: float | None = None,
) -> plt.Figure:
    """Density snapshot plot for two competing species.

    Species 1 (u) drawn solid; species 2 (v) drawn dashed.
    Both species share colour per snapshot time.
    """
    snap_times = sorted(set(res["snapshots_u"]) | set(res["snapshots_v"]))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(snap_times), 1)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ts in enumerate(snap_times):
        c = colors[i]
        if ts in res["snapshots_u"]:
            ax.plot(res["s"], res["snapshots_u"][ts], color=c, ls="-",
                    linewidth=2.0, label=f"u (species 1)  τ={ts:g}")
        if ts in res["snapshots_v"]:
            ax.plot(res["s"], res["snapshots_v"][ts], color=c, ls="--",
                    linewidth=2.0, label=f"v (species 2)  τ={ts:g}")

    if K1 is not None:
        ax.axhline(K1, color="steelblue", ls=":", lw=1.2, label=f"K1={K1:g}")
    if K2 is not None:
        ax.axhline(K2, color="tomato",    ls=":", lw=1.2, label=f"K2={K2:g}")
    for bnd_label, bnd_pos, bnd_color, bnd_ls in _extract_boundaries(res):
        ax.axvline(
            bnd_pos, color=bnd_color, ls=bnd_ls, lw=1.2,
            label=f"{bnd_label} = {bnd_pos:.3f}",
        )

    #ax.set_title(f"Two-Species Density Snapshots — {title}")
    ax.set_xlabel("Dimensionless position ξ")
    ax.set_ylabel("Dimensionless density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_biomass_2s(res: dict, title: str) -> plt.Figure:
    """Biomass time-series for two competing species in side-by-side panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = _two_species_palette()
    species_panels = [
        (
            axes[0],
            "Species 1",
            res["B1_tot"],
            res["B1_in"],
            res["B1_out"],
            palette["sp1_tot"],
            palette["sp1_in"],
            palette["sp1_out"],
        ),
        (
            axes[1],
            "Species 2",
            res["B2_tot"],
            res["B2_in"],
            res["B2_out"],
            palette["sp2_tot"],
            palette["sp2_in"],
            palette["sp2_out"],
        ),
    ]

    for ax, species_label, total, inside, outside, c_tot, c_in, c_out in species_panels:
        ax.plot(res["time"], total, color=c_tot, lw=2.1, ls="-",
                label=f"{species_label} total")
        ax.plot(res["time"], inside, color=c_in, lw=1.6, ls="--",
                label=f"{species_label} inside EEZ")
        ax.plot(res["time"], outside, color=c_out, lw=1.6, ls=":",
                label=f"{species_label} outside EEZ")
        _add_harvest_start_marker(ax, res)
        ax.set_title(species_label)
        ax.set_xlabel("Dimensionless time τ")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Dimensionless biomass")
    fig.tight_layout()
    return fig


def _clean_2s_case_label(label: str) -> str:
    """Normalise comparison-case labels for display."""
    clean_label = re.sub(r"^[A-Za-z]\([^)]*\):\s*", "", label)
    clean_label = re.sub(r"\bu\b", "x", clean_label)
    clean_label = re.sub(r"\bv\b", "y", clean_label)
    return clean_label


def print_biomass_comparison_2s_eez_shares(
    cases: list[tuple[dict, str]],
) -> None:
    """Print final inside/outside EEZ biomass shares for each comparison case."""
    for res, label in cases:
        clean_label = _clean_2s_case_label(label)

        b1_total = float(np.asarray(res["B1_tot"])[-1])
        b2_total = float(np.asarray(res["B2_tot"])[-1])
        b1_in_pct = 0.0 if np.isclose(b1_total, 0.0) else 100.0 * float(np.asarray(res["B1_in"])[-1]) / b1_total
        b1_out_pct = 0.0 if np.isclose(b1_total, 0.0) else 100.0 * float(np.asarray(res["B1_out"])[-1]) / b1_total
        b2_in_pct = 0.0 if np.isclose(b2_total, 0.0) else 100.0 * float(np.asarray(res["B2_in"])[-1]) / b2_total
        b2_out_pct = 0.0 if np.isclose(b2_total, 0.0) else 100.0 * float(np.asarray(res["B2_out"])[-1]) / b2_total

        print(f"{clean_label}:")
        print(f"  x: {b1_in_pct:.1f}% inside EEZ, {b1_out_pct:.1f}% outside EEZ")
        print(f"  y: {b2_in_pct:.1f}% inside EEZ, {b2_out_pct:.1f}% outside EEZ")
        print()


def plot_biomass_comparison_2s_eez_percentages(
    cases: list[tuple[dict, str]],
    title: str | None = None,
) -> plt.Figure:
    """Plot final inside/outside EEZ biomass shares across start-position cases."""
    if not cases:
        raise ValueError("At least one case is required for EEZ percentage plotting.")

    font_size = 13
    palette = _two_species_palette()
    case_labels: list[str] = []
    sp1_in_pct: list[float] = []
    sp1_out_pct: list[float] = []
    sp2_in_pct: list[float] = []
    sp2_out_pct: list[float] = []

    for res, label in cases:
        clean_label = _clean_2s_case_label(label)
        case_labels.append(clean_label)

        b1_total = float(np.asarray(res["B1_tot"])[-1])
        b2_total = float(np.asarray(res["B2_tot"])[-1])
        sp1_in_pct.append(
            0.0 if np.isclose(b1_total, 0.0) else 100.0 * float(np.asarray(res["B1_in"])[-1]) / b1_total
        )
        sp1_out_pct.append(
            0.0 if np.isclose(b1_total, 0.0) else 100.0 * float(np.asarray(res["B1_out"])[-1]) / b1_total
        )
        sp2_in_pct.append(
            0.0 if np.isclose(b2_total, 0.0) else 100.0 * float(np.asarray(res["B2_in"])[-1]) / b2_total
        )
        sp2_out_pct.append(
            0.0 if np.isclose(b2_total, 0.0) else 100.0 * float(np.asarray(res["B2_out"])[-1]) / b2_total
        )

    x = np.arange(len(case_labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    species_panels = [
        ("x", sp1_in_pct, sp1_out_pct, palette["sp1_in"], palette["sp1_out"]),
        ("y", sp2_in_pct, sp2_out_pct, palette["sp2_in"], palette["sp2_out"]),
    ]

    for ax, (species_label, inside_pct, outside_pct, c_in, c_out) in zip(axes, species_panels):
        ax.bar(x, inside_pct, width=0.65, color=c_in, label="Inside EEZ")
        ax.bar(x, outside_pct, width=0.65, bottom=inside_pct, color=c_out, label="Outside EEZ")
        ax.set_title(species_label, fontsize=font_size)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=font_size, rotation=15, ha="right")
        ax.set_xlabel("Initial positions", fontsize=font_size)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="y", labelsize=font_size)

    axes[0].set_ylabel("Final biomass share (%)", fontsize=font_size)
    if title:
        fig.suptitle(title, fontsize=font_size)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_biomass_2s_eez_percentages_by_start_position(
    start_positions: np.ndarray,
    x_in_pct: np.ndarray,
    x_out_pct: np.ndarray,
    y_in_pct: np.ndarray,
    y_out_pct: np.ndarray,
    title: str | None = None,
    eez_boundary: float | None = None,
    x_label: str = "Initial center ξ",
) -> plt.Figure:
    """Plot final EEZ biomass shares against initial start position."""
    start_positions = np.asarray(start_positions, dtype=float)
    x_in_pct = np.asarray(x_in_pct, dtype=float)
    x_out_pct = np.asarray(x_out_pct, dtype=float)
    y_in_pct = np.asarray(y_in_pct, dtype=float)
    y_out_pct = np.asarray(y_out_pct, dtype=float)

    if not (
        start_positions.shape
        == x_in_pct.shape
        == x_out_pct.shape
        == y_in_pct.shape
        == y_out_pct.shape
    ):
        raise ValueError("All start-position percentage arrays must share the same shape.")

    font_size = 13
    palette = _two_species_palette()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    species_panels = [
        ("x", x_in_pct, x_out_pct, palette["sp1_in"], palette["sp1_out"]),
        ("y", y_in_pct, y_out_pct, palette["sp2_in"], palette["sp2_out"]),
    ]

    for idx, (ax, (species_label, inside_pct, outside_pct, c_in, c_out)) in enumerate(zip(axes, species_panels)):
        ax.plot(start_positions, inside_pct, color=c_in, lw=2.0, label="Inside EEZ")
        ax.plot(start_positions, outside_pct, color=c_out, lw=2.0, label="Outside EEZ")
        if eez_boundary is not None:
            ax.axvline(
                eez_boundary,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="EEZ boundary" if idx == 0 else None,
            )
        ax.set_title(species_label, fontsize=font_size)
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="both", labelsize=font_size)

    axes[0].set_ylabel("Final biomass share (%)", fontsize=font_size)
    if title:
        fig.suptitle(title, fontsize=font_size)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_biomass_comparison_2s(
    cases: list[tuple[dict, str]],
    title: str | None = None,
) -> plt.Figure:
    """Compare harvested biomass across cases in one figure with case rows."""
    if not cases:
        raise ValueError("At least one case is required for comparison plotting.")

    font_size = 13
    palette = _two_species_palette()
    species_configs = [
        (
            "Species 1",
            ("B1_in", "B1_out", "B1_tot"),
            (palette["sp1_in"], palette["sp1_out"], palette["sp1_tot"]),
            ("inside EEZ", "outside EEZ", "total"),
        ),
        (
            "Species 2",
            ("B2_in", "B2_out", "B2_tot"),
            (palette["sp2_in"], palette["sp2_out"], palette["sp2_tot"]),
            ("inside EEZ", "outside EEZ", "total"),
        ),
    ]

    n_cases = len(cases)
    fig, axes = plt.subplots(
        n_cases,
        2,
        figsize=(13, 3.5 * n_cases),
        sharex=True,
        sharey="row",
    )
    if n_cases == 1:
        axes = np.asarray([axes])

    for row, (res, lbl) in enumerate(cases):
        clean_label = _clean_2s_case_label(lbl)
        for col, (species_label, series_keys, colours, curve_labels) in enumerate(species_configs):
            ax = axes[row, col]
            for key, colour, curve_label in zip(series_keys, colours, curve_labels):
                ax.plot(
                    res["time"],
                    res[key],
                    color=colour,
                    lw=2.0 if curve_label == "total" else 1.6,
                    label=f"{species_label} {curve_label}",
                )

            _add_harvest_start_marker(ax, res)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            ax.set_title(f"{clean_label} - {species_label}", fontsize=font_size)
            ax.tick_params(axis="both", labelsize=font_size)

        axes[row, 0].set_ylabel("Dimensionless biomass", fontsize=font_size)

    axes[-1, 0].set_xlabel("Dimensionless time \u03c4", fontsize=font_size)
    axes[-1, 1].set_xlabel("Dimensionless time \u03c4", fontsize=font_size)
    if title:
        fig.suptitle(title, fontsize=font_size)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    return fig


def plot_catch_2s(res: dict, title: str) -> plt.Figure:
    """Catch rate + cumulative catch for two competing species (twin axes)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    palette = _two_species_palette()

    ax1.plot(res["time"], res["Catch1"], color=palette["sp1_tot"], lw=2.0, label="Catch rate sp.1")
    ax1.plot(res["time"], res["Catch2"], color=palette["sp2_tot"], lw=2.0, label="Catch rate sp.2")
    _add_harvest_start_marker(ax1, res)
    ax1.set_xlabel("Dimensionless time τ")
    ax1.set_ylabel("Dimensionless catch rate")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(res["time"], res["CumCatch1"], color=palette["sp1_in"], lw=1.5, ls="--",
             label="Cum. catch sp.1")
    ax2.plot(res["time"], res["CumCatch2"], color=palette["sp2_in"], lw=1.5, ls="--",
             label="Cum. catch sp.2")
    ax2.set_ylabel("Cumulative dimensionless catch")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax1.set_title(f"Two-Species Catch — {title}")
    fig.tight_layout()
    return fig


def plot_2d_snapshots_5(res: dict, title: str, K: float, y_boundary: float, cmap: str = 'viridis') -> None:
    """Heatmap grid: one panel per snapshot time.

    y=0 (coast) is at the bottom of each panel.
    Red dashed horizontal line marks the 200-mile EEZ boundary.
    """
    snap_times_sorted = sorted(res['snapshots'].keys())
    n = len(snap_times_sorted)
    x_grid = res['x_grid']
    y_grid = res['y_grid']

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5),
                             sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    vmin, vmax_c = 0.0, K * 1.1

    for ax, ts in zip(axes, snap_times_sorted):
        U = res['snapshots'][ts]
        pcm = ax.pcolormesh(x_grid, y_grid, U,
                            cmap=cmap, vmin=vmin, vmax=vmax_c,
                            shading='auto')
        ax.axhline(y_boundary, color='red', ls='--', lw=1.5,
                   label='200-mile EEZ')
        ax.set_title(f't = {ts:g} yr', fontsize=10)
        ax.set_xlabel('x  (along-shore, miles)')
        if ax is axes[0]:
            ax.set_ylabel('y  (offshore, miles)')
            ax.legend(fontsize=8, loc='upper right')
        fig.colorbar(pcm, ax=ax, label='density u')

    fig.suptitle(f'§5  {title}', fontsize=11, y=1.01)
    plt.show()


def plot_2d_biomass_5(res: dict, title: str) -> None:
    """Biomass time series for §5: total, inside EEZ, outside EEZ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['time'], res['Btot'], lw=2.0, color='steelblue',
            label='Total biomass')
    ax.plot(res['time'], res['Bin'],  lw=1.5, ls='--', color='seagreen',
            label='Inside EEZ')
    ax.plot(res['time'], res['Bout'], lw=1.5, ls=':',  color='tomato',
            label='Outside EEZ')
    ax.set_xlabel('Time  (yr)')
    ax.set_ylabel('Biomass  (miles\u00b2 \u00d7 density)')
    ax.set_title(f'\u00a75 Biomass \u2014 {title}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_2d_two_species(res: dict, title: str, K1: float, K2: float, y_boundary: float) -> None:
    """For each snapshot: three side-by-side panels \u2014 U, V, U+V overlay.

    Row layout: one row per snapshot time.
    Panel 1: U density heatmap (Blues).
    Panel 2: V density heatmap (Reds).
    Panel 3: U heatmap (Blues) with V contour lines (red) overlaid.
    """
    snap_times_sorted = sorted(res['snapshots_U'].keys())
    n_snaps = len(snap_times_sorted)
    x_grid  = res['x_grid']
    y_grid  = res['y_grid']

    fig, axes = plt.subplots(n_snaps, 3,
                             figsize=(15, 3.5 * n_snaps),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    if n_snaps == 1:
        axes = axes[np.newaxis, :]

    for row, ts in enumerate(snap_times_sorted):
        U = res['snapshots_U'][ts]
        V = res['snapshots_V'][ts]

        ax = axes[row, 0]
        pcm = ax.pcolormesh(x_grid, y_grid, U,
                            cmap='Blues', vmin=0.0, vmax=K1 * 1.05,
                            shading='auto')
        ax.axhline(y_boundary, color='red', ls='--', lw=1.5, label='200-mile EEZ')
        ax.set_title(f'U  (species 1)   t = {ts:g} yr', fontsize=9)
        ax.set_ylabel('y  (offshore, miles)')
        if row == 0:
            ax.legend(fontsize=7, loc='upper right')
        fig.colorbar(pcm, ax=ax, label='density u')

        ax = axes[row, 1]
        pcm = ax.pcolormesh(x_grid, y_grid, V,
                            cmap='Reds', vmin=0.0, vmax=K2 * 1.05,
                            shading='auto')
        ax.axhline(y_boundary, color='red', ls='--', lw=1.5)
        ax.set_title(f'V  (species 2)   t = {ts:g} yr', fontsize=9)
        ax.set_yticklabels([])
        fig.colorbar(pcm, ax=ax, label='density v')

        ax = axes[row, 2]
        pcm = ax.pcolormesh(x_grid, y_grid, U,
                            cmap='Blues', vmin=0.0, vmax=K1 * 1.05,
                            shading='auto', alpha=0.85)
        fig.colorbar(pcm, ax=ax, label='density u')
        vmax_v = V.max() if V.max() > 1e-6 else 1.0
        levels_v = np.linspace(0.05 * vmax_v, vmax_v, 6)
        cs = ax.contour(x_grid, y_grid, V,
                        levels=levels_v, colors='red', linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        ax.axhline(y_boundary, color='red', ls='--', lw=1.5, label='200-mile EEZ')
        ax.set_title(f'U + V overlay   t = {ts:g} yr', fontsize=9)
        ax.set_yticklabels([])
        if row == 0:
            ax.legend(fontsize=7, loc='upper right')

    for col in range(3):
        axes[-1, col].set_xlabel('x  (along-shore, miles)')

    fig.suptitle(f'\u00a76  {title}', fontsize=11, y=1.01)
    plt.show()


def plot_2d_biomass_6(res: dict, title: str) -> None:
    """Biomass time series for \u00a76: both species, total and EEZ-split."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(res['time'], res['B1tot'], color='steelblue', lw=2.0,
            label='U total (species 1)')
    ax.plot(res['time'], res['B1in'],  color='steelblue', lw=1.2, ls='--',
            label='U inside EEZ')
    ax.plot(res['time'], res['B1out'], color='steelblue', lw=1.2, ls=':',
            label='U outside EEZ')
    ax.plot(res['time'], res['B2tot'], color='tomato', lw=2.0,
            label='V total (species 2)')
    ax.plot(res['time'], res['B2in'],  color='tomato', lw=1.2, ls='--',
            label='V inside EEZ')
    ax.plot(res['time'], res['B2out'], color='tomato', lw=1.2, ls=':',
            label='V outside EEZ')
    ax.set_xlabel('Time  (yr)')
    ax.set_ylabel('Biomass  (miles\u00b2 \u00d7 density)')
    ax.set_title(f'\u00a76 Biomass \u2014 {title}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    plt.show()


def plot_2d_biomass_comparison_6(
    results: dict[str, dict],
    title: str,
    scenario_labels: dict[str, str] | None = None,
) -> None:
    """Combined §6 biomass comparison across scenarios."""
    qty_grid = [
        [('B1tot', 'Total biomass'), ('B1in', 'Inside EEZ'), ('B1out', 'Outside EEZ')],
        [('B2tot', 'Total biomass'), ('B2in', 'Inside EEZ'), ('B2out', 'Outside EEZ')],
    ]
    row_labels = ['U biomass  (miles² × density)', 'V biomass  (miles² × density)']
    default_colors = {'B0': 'seagreen', 'B1': 'goldenrod', 'B2': 'tomato'}
    fallback_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                             sharex=True, sharey='row',
                             constrained_layout=True)

    for row, qty_row in enumerate(qty_grid):
        for col, (qty, panel_title) in enumerate(qty_row):
            ax = axes[row, col]
            for idx, (key, res) in enumerate(results.items()):
                color = default_colors.get(
                    key,
                    fallback_colors[idx % len(fallback_colors)] if fallback_colors else None,
                )
                label = scenario_labels[key] if scenario_labels and key in scenario_labels else key
                ax.plot(res['time'], res[qty], lw=1.8, color=color, label=label)
            if row == 0:
                ax.set_title(panel_title)
            ax.set_xlabel('Time  (yr)')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_labels[row])
                ax.legend(fontsize=8)

    #fig.suptitle(f'§6 Comparison — {title}', fontsize=11, y=1.01)
    plt.show()


# ---------------------------------------------------------------------------
# Two-species competing plots (Section 4)
# ---------------------------------------------------------------------------

def plot_heatmap_2s(
    res: dict,
    title: str,
) -> plt.Figure:
    """Side-by-side heatmaps of the two-species density fields u and v."""
    if "u_full" not in res or "v_full" not in res:
        raise ValueError("store_full=True required for plot_heatmap_2s")

    s_arr = res["s"]
    t_arr = res["t_full"]
    u_full = res["u_full"]
    v_full = res["v_full"]
    boundaries = _extract_boundaries(res)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, field, sp_label in zip(axes, [u_full, v_full], ["u  (species 1)", "v  (species 2)"]):
        pcm = ax.pcolormesh(s_arr, t_arr, field, cmap="viridis", shading="auto")
        for bnd_label, bnd_pos, bnd_color, bnd_ls in boundaries:
            ax.axvline(
                bnd_pos, color=bnd_color, ls=bnd_ls, lw=1.5,
                label=f"{bnd_label} = {bnd_pos:.3f}",
            )
        ax.set_xlabel("Dimensionless position ξ", fontsize = 13)
        #ax.set_title(f"{sp_label}")
        if boundaries:
            ax.legend(fontsize=13)
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("Dimensionless density")

    axes[0].set_ylabel("Dimensionless time τ", fontsize = 13)
    fig.tight_layout()
    return fig


def _plot_biomass_comparison_2s_legacy(
    cases: list[tuple[dict, str]],
    title: str | None = None,
) -> plt.Figure:
    """Compare harvested two-species biomass trajectories across multiple cases."""
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), sharey=True)
    if len(cases) == 1:
        axes = [axes]

    palette = _two_species_palette()
    for idx, (ax, (res, lbl)) in enumerate(zip(axes, cases)):
        ax.plot(res["time"], res["B1_in"], color=palette["sp1_in"], lw=1.6,
                label="Species 1 inside EEZ")
        ax.plot(res["time"], res["B1_out"], color=palette["sp1_out"], lw=1.6,
                label="Species 1 outside EEZ")
        ax.plot(res["time"], res["B1_tot"], color=palette["sp1_tot"], lw=2.0,
                label="Species 1 total")
        ax.plot(res["time"], res["B2_in"], color=palette["sp2_in"], lw=1.6,
                label="Species 2 inside EEZ")
        ax.plot(res["time"], res["B2_out"], color=palette["sp2_out"], lw=1.6,
                label="Species 2 outside EEZ")
        ax.plot(res["time"], res["B2_tot"], color=palette["sp2_tot"], lw=2.0,
                label="Species 2 total")

        has_catch = (
            np.any(np.asarray(res["Catch1"]) > 0.0)
            or np.any(np.asarray(res["Catch2"]) > 0.0)
        )
        if has_catch:
            ax.axvline(
                res["harvest_start_time"],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"Harvest starts (τ={res['harvest_start_time']:g})" if idx == 0 else None,
            )

        #ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Dimensionless time τ")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel("Dimensionless biomass")
    fig.tight_layout()
    return fig
