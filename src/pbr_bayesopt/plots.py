import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_reference_map(lights, dilution_rates, values, optimum, label, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    contour = ax.contourf(dilution_rates, lights, values, levels=20, cmap="viridis")
    ax.plot(optimum[1], optimum[0], marker="*", color="white", markersize=14, markeredgecolor="black", linestyle="none", label="Optimum")
    ax.set_xlabel("Dilution rate (day⁻¹)")
    ax.set_ylabel("Emitted PPFD (µmol m⁻² s⁻¹)")
    ax.legend(loc="upper right")
    fig.colorbar(contour, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_search(lights, dilution_rates, values, history, n_initial, label, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    contour = ax.contourf(dilution_rates, lights, values, levels=20, cmap="viridis", alpha=0.85)
    points = history.points
    ax.scatter(points[:n_initial, 1], points[:n_initial, 0], facecolor="white", edgecolor="black", s=36, label="Initial design")
    ax.scatter(points[n_initial:, 1], points[n_initial:, 0], c=np.arange(len(points) - n_initial), cmap="autumn_r", edgecolor="black", s=36, label="Bayesian optimisation (later = redder)")
    best = history.best_point
    ax.plot(best[1], best[0], marker="*", color="white", markersize=14, markeredgecolor="black", linestyle="none", label="Best found")
    ax.set_xlabel("Dilution rate (day⁻¹)")
    ax.set_ylabel("Emitted PPFD (µmol m⁻² s⁻¹)")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(contour, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regret(curves, optimum_value, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, runs in curves.items():
        regret = 100 * (optimum_value - np.asarray(runs)) / optimum_value
        regret = np.maximum(regret, 1e-3)
        median = np.median(regret, axis=0)
        lower, upper = np.percentile(regret, [25, 75], axis=0)
        evaluations = np.arange(1, len(median) + 1)
        ax.plot(evaluations, median, label=name)
        ax.fill_between(evaluations, lower, upper, alpha=0.25)
    ax.set_yscale("log")
    ax.set_xlabel("Reactor simulations")
    ax.set_ylabel("Gap to true optimum (%)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
