"""Article-style showcase of the neural-mass simulation models.

Run from the repository root with::

    uv run python scripts/examples/neural_mass_showcase.py --output figures/neural_mass_showcase.pdf

The figure contains: Hopf bifurcation and phase portraits, Wilson--Cowan
population dynamics, and Kuramoto synchronization under weak and strong
coupling.  The output is vector PDF by default and can also be saved as PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from pyeeg.simulate import HopfOscillator, Kuramoto, WilsonCowan

COLORS = {
    "ink": "#17202A",
    "muted": "#667085",
    "grid": "#D9E2EC",
    "blue": "#2F6BFF",
    "teal": "#00A6A6",
    "orange": "#F28E2B",
    "coral": "#E45756",
    "purple": "#7A5195",
    "green": "#59A14F",
}


def style_axes(ax: plt.Axes, *, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="semibold", color=COLORS["ink"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], length=3)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)


def hopf_panel(ax_bif: plt.Axes, ax_phase: plt.Axes, ax_trace: plt.Axes) -> None:
    """Show the supercritical Hopf transition and a representative orbit."""
    # For z' = (a + i*omega - |z|^2)z, the stable cycle has radius
    # sqrt(a) for a > 0.  Overlay short numerical runs initialized on that
    # cycle to make the bifurcation relation explicit and reproducible.
    amplitudes = []
    a_values = np.linspace(-0.12, 0.20, 33)
    for a in a_values:
        model = HopfOscillator(a=a, frequency=6.0, dt=0.002, seed=10)
        radius = np.sqrt(max(a, 0.0))
        states, _ = model.simulate(x0=np.array([radius, 0.0]), tmax=0.8)
        amplitudes.append(np.mean(np.hypot(states[-100:, 0], states[-100:, 1])))

    ax_bif.plot(
        a_values,
        np.sqrt(np.maximum(a_values, 0)),
        color=COLORS["muted"],
        linestyle="--",
        linewidth=1.2,
        label=r"theory $\sqrt{a_+}$",
    )
    ax_bif.scatter(
        a_values,
        amplitudes,
        color=COLORS["blue"],
        s=15,
        zorder=3,
        label="Euler simulation",
    )
    ax_bif.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax_bif.set_xlabel(r"Bifurcation parameter $a$")
    ax_bif.set_ylabel(r"Late-time amplitude $|z|$")
    ax_bif.legend(frameon=False, fontsize=8, loc="lower right")
    ax_bif.text(
        0.04,
        0.18,
        "stable fixed point",
        transform=ax_bif.transAxes,
        color=COLORS["muted"],
        fontsize=8,
    )
    ax_bif.text(
        0.62,
        0.18,
        "limit cycle",
        transform=ax_bif.transAxes,
        color=COLORS["blue"],
        fontsize=8,
    )
    style_axes(ax_bif, title="Hopf oscillator · bifurcation")

    model = HopfOscillator(a=0.12, frequency=6.0, dt=0.001, seed=2)
    states, _ = model.simulate(x0=np.array([np.sqrt(0.12), 0.0]), tmax=1.2)
    time = np.arange(len(states)) * model.dt
    ax_phase.plot(states[:, 0], states[:, 1], color=COLORS["purple"], linewidth=1.4, alpha=0.5)
    ax_phase.scatter(
        states[:, 0], states[:, 1], c=time, cmap="viridis", s=4, zorder=3,
    )
    ax_phase.scatter(states[0, 0], states[0, 1], color=COLORS["orange"], s=30, zorder=4, label="start")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_aspect("equal", adjustable="box")
    style_axes(ax_phase, title="Hopf oscillator · limit cycle")

    ax_trace.plot(time, states[:, 0], color=COLORS["blue"], linewidth=1.2, label="x")
    ax_trace.plot(time, states[:, 1], color=COLORS["coral"], linewidth=1.2, label="y")
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("State")
    ax_trace.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax_trace, title="Hopf oscillator · node dynamics")


def wilson_cowan_panel(ax_trace: plt.Axes, ax_phase: plt.Axes) -> None:
    """Show transient excitatory/inhibitory population dynamics."""
    model = WilsonCowan(dt=0.001, tau_e=0.012, tau_i=0.018, P=1.25, seed=4)
    states, _ = model.simulate(tmax=1.5)
    time = np.arange(len(states)) * model.dt
    ax_trace.plot(time, states[:, 0], color=COLORS["orange"], linewidth=1.5, label="E")
    ax_trace.plot(time, states[:, 1], color=COLORS["teal"], linewidth=1.5, label="I")
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Population activity")
    ax_trace.set_ylim(bottom=0)
    ax_trace.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax_trace, title="Wilson–Cowan · E/I populations")

    # Endpoint markers show the direction without adding another colorbar.
    ax_phase.plot(states[:, 0], states[:, 1], color=COLORS["green"], linewidth=1.2)
    ax_phase.scatter(
        states[0, 0], states[0, 1], color=COLORS["orange"], s=30,
        zorder=3, label="start",
    )
    ax_phase.scatter(
        states[-1, 0], states[-1, 1], color=COLORS["ink"], s=22,
        zorder=3, label="end",
    )
    ax_phase.set_xlabel("Excitatory activity E")
    ax_phase.set_ylabel("Inhibitory activity I")
    ax_phase.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax_phase, title="Wilson–Cowan · population trajectory")


def kuramoto_panel(ax_raster: plt.Axes, ax_order: plt.Axes) -> None:
    """Compare incoherent and synchronized phase-network regimes."""
    n_nodes = 24
    weights = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    frequencies = np.linspace(8.5, 11.5, n_nodes)
    initial_phases = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    traces = []
    orders = []
    dt = 0.002
    for strength in (0.0, 1.8):
        network = Kuramoto(
            N=n_nodes,
            W=weights,
            frequency=10.0,
            dt=dt,
            coupling_strength=strength,
            seed=8,
        )
        # Introduce frequency heterogeneity before integration.
        for node, frequency in zip(network.nodes, frequencies, strict=False):
            node.frequency = frequency
            node.omega = 2 * np.pi * frequency
        for node, phase in zip(network.nodes, initial_phases, strict=False):
            node.x[0] = phase

        n_samples = int(2.0 / dt)
        output = np.zeros((n_samples, n_nodes))
        phase_history = np.zeros((n_samples, n_nodes))
        phase_history[0] = [node.x[0] for node in network.nodes]
        output[0] = [node.read_out() for node in network.nodes]
        for sample in range(1, n_samples):
            network.step()
            phase_history[sample] = [node.x[0] for node in network.nodes]
            output[sample] = [node.read_out() for node in network.nodes]
        traces.append(output)
        orders.append(np.abs(np.mean(np.exp(1j * phase_history), axis=1)))

    image = np.vstack([traces[0].T, traces[1].T])
    cmap = LinearSegmentedColormap.from_list(
        "phase_readout", ["#F4F7FB", COLORS["blue"]]
    )
    im = ax_raster.imshow(
        image,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[0, 2, 0, 2 * n_nodes],
        interpolation="nearest",
    )
    cbar = ax_raster.figure.colorbar(im, ax=ax_raster, fraction=0.046, pad=0.02)
    cbar.set_label("sin(θ)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors=COLORS["muted"])
    ax_raster.axhline(n_nodes, color="white", linewidth=1.2)
    ax_raster.set_yticks([n_nodes * 0.5, n_nodes * 1.5])
    ax_raster.set_yticklabels(["K = 0 · weak", "K = 1.8 · strong"])
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("")
    style_axes(ax_raster, title="Kuramoto · network readouts")

    time = np.arange(len(orders[0])) * dt
    ax_order.plot(time, orders[0], color=COLORS["muted"], linewidth=1.5, label="K = 0")
    ax_order.plot(time, orders[1], color=COLORS["blue"], linewidth=1.8, label="K = 1.8")
    ax_order.set_xlabel("Time (s)")
    ax_order.set_ylabel(r"Kuramoto order parameter $r$")
    ax_order.set_ylim(0, 1)
    ax_order.legend(frameon=False, loc="lower right")
    style_axes(ax_order, title="Kuramoto · ordering tendency")


def make_figure(layout: str = "v5") -> plt.Figure:
    """Create a compact four-row figure with balanced, readable panels."""
    if layout != "v5":
        raise ValueError("Only the publication layout 'v5' is supported")
    fig = plt.figure(figsize=(12.0, 10.0), constrained_layout=False)
    fig.suptitle(
        "Neural-mass dynamics: local regimes and network organization",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.03,
        0.952,
        "Deterministic textbook examples with Euler integration",
        color=COLORS["muted"],
        fontsize=10,
    )
    grid = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.0, 1.0, 1.0, 0.9],
        hspace=0.50,
        wspace=0.38,
        left=0.08,
        right=0.96,
        bottom=0.08,
        top=0.88,
    )
    hopf_panel(
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2:]),
        fig.add_subplot(grid[1, :2]),
    )
    wilson_cowan_panel(fig.add_subplot(grid[1, 2:]), fig.add_subplot(grid[2, :2]))
    kuramoto_panel(fig.add_subplot(grid[2, 2:]), fig.add_subplot(grid[3, :]))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (.pdf, .png, or another Matplotlib format).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="v5",
        choices=["v5"],
        help="Figure layout variant.",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    figure = make_figure(layout=args.layout)
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output)
        print(f"Saved {args.output}")
        plt.close(figure)


if __name__ == "__main__":
    main()
