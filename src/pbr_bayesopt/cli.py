import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from pbr_bayesopt.objective import TARGETS, ReactorObjective
from pbr_bayesopt.optimiser import bayesian_optimisation, random_search
from pbr_bayesopt.plots import plot_reference_map, plot_regret, plot_search

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def reference_path(target):
    return os.path.join(RESULTS_DIR, f"reference_{target}.json")


def load_reference(target):
    path = reference_path(target)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run `pbr-bayesopt reference --target {target}` first.")
    with open(path) as f:
        reference = json.load(f)
    return {key: np.array(value) if isinstance(value, list) else value for key, value in reference.items()}


def run_reference(args):
    objective = ReactorObjective(target=args.target)
    lights = np.linspace(*objective.bounds["light"], args.grid)
    dilution_rates = np.linspace(*objective.bounds["dilution_rate"], args.grid)

    start = time.time()
    values = np.array([[objective((light, dilution)) for dilution in dilution_rates] for light in lights])
    i, j = np.unravel_index(np.argmax(values), values.shape)

    polished = minimize(
        lambda point: -objective(point),
        x0=[lights[i], dilution_rates[j]],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-7},
    )
    optimum = np.clip(polished.x, objective.lower, objective.upper)
    optimum_value = float(-polished.fun)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(reference_path(args.target), "w") as f:
        json.dump(
            {
                "target": args.target,
                "lights": lights.tolist(),
                "dilution_rates": dilution_rates.tolist(),
                "values": values.tolist(),
                "optimum": optimum.tolist(),
                "optimum_value": optimum_value,
                "n_simulations": int(values.size + polished.nfev),
            },
            f,
            indent=2,
        )
    plot_reference_map(lights, dilution_rates, values, optimum, objective.label, os.path.join(FIGURES_DIR, f"reference_{args.target}.png"))

    print(f"Grid of {args.grid}x{args.grid} plus {polished.nfev} polishing steps in {time.time() - start:.0f}s")
    print(f"Optimum: light {optimum[0]:.0f}, dilution rate {optimum[1]:.3f}, {objective.label} {optimum_value:.4f}")


def run_optimise(args):
    objective = ReactorObjective(target=args.target)
    reference = load_reference(args.target)

    def report(iteration, point, value, model):
        print(f"{iteration + 1:3d}  light {point[0]:6.0f}  dilution {point[1]:.3f}  value {value:.4f}")

    history = bayesian_optimisation(
        objective, objective.lower, objective.upper, n_initial=args.initial, n_iterations=args.iterations, seed=args.seed, callback=report
    )
    gap = 100 * (reference["optimum_value"] - history.best_value) / reference["optimum_value"]
    print(f"\nBest after {len(history.values)} simulations: light {history.best_point[0]:.0f}, dilution rate {history.best_point[1]:.3f}")
    print(f"{objective.label}: {history.best_value:.4f} ({gap:.2f}% below the reference optimum)")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_search(
        reference["lights"], reference["dilution_rates"], reference["values"], history, args.initial, objective.label,
        os.path.join(FIGURES_DIR, f"search_{args.target}.png"),
    )


def run_compare(args):
    objective = ReactorObjective(target=args.target)
    reference = load_reference(args.target)
    budget = args.initial + args.iterations

    curves = {"Bayesian optimisation": [], "Random search": []}
    for seed in range(args.seeds):
        start = time.time()
        bo = bayesian_optimisation(objective, objective.lower, objective.upper, n_initial=args.initial, n_iterations=args.iterations, seed=seed)
        rs = random_search(objective, objective.lower, objective.upper, n_evaluations=budget, seed=seed)
        curves["Bayesian optimisation"].append(bo.best_so_far.tolist())
        curves["Random search"].append(rs.best_so_far.tolist())
        print(f"seed {seed}: Bayesian optimisation {bo.best_value:.4f}, random search {rs.best_value:.4f} ({time.time() - start:.0f}s)")

    optimum_value = reference["optimum_value"]
    summary = {"target": args.target, "budget": budget, "seeds": args.seeds, "optimum_value": optimum_value, "curves": curves}
    for name, runs in curves.items():
        final_gap = 100 * (optimum_value - np.array(runs)[:, -1]) / optimum_value
        within_one_percent = [int(np.argmax(np.array(run) >= 0.99 * optimum_value)) + 1 if run[-1] >= 0.99 * optimum_value else None for run in runs]
        reached = [n for n in within_one_percent if n is not None]
        summary[name] = {
            "median_final_gap_percent": float(np.median(final_gap)),
            "runs_within_one_percent": len(reached),
            "median_simulations_to_one_percent": float(np.median(reached)) if reached else None,
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"comparison_{args.target}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_regret(curves, optimum_value, os.path.join(FIGURES_DIR, f"regret_{args.target}.png"))

    print(f"\n{'Method':24s} {'Median final gap':>18s} {'Runs within 1%':>16s} {'Median sims to 1%':>19s}")
    for name in curves:
        s = summary[name]
        sims = "-" if s["median_simulations_to_one_percent"] is None else f"{s['median_simulations_to_one_percent']:.0f}"
        print(f"{name:24s} {s['median_final_gap_percent']:17.2f}% {s['runs_within_one_percent']:>11d} / {args.seeds:<2d} {sims:>19s}")


def main():
    parser = argparse.ArgumentParser(prog="pbr-bayesopt", description="Bayesian optimisation of a photobioreactor model")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(command):
        command.add_argument("--target", choices=sorted(TARGETS), default="energy_yield")

    reference = commands.add_parser("reference", help="map the objective on a grid and locate the true optimum")
    add_common(reference)
    reference.add_argument("--grid", type=int, default=25)
    reference.set_defaults(run=run_reference)

    optimise = commands.add_parser("optimise", help="run one Bayesian optimisation and plot where it sampled")
    add_common(optimise)
    optimise.add_argument("--initial", type=int, default=8)
    optimise.add_argument("--iterations", type=int, default=22)
    optimise.add_argument("--seed", type=int, default=0)
    optimise.set_defaults(run=run_optimise)

    compare = commands.add_parser("compare", help="compare Bayesian optimisation with random search over several seeds")
    add_common(compare)
    compare.add_argument("--initial", type=int, default=8)
    compare.add_argument("--iterations", type=int, default=22)
    compare.add_argument("--seeds", type=int, default=10)
    compare.set_defaults(run=run_compare)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
