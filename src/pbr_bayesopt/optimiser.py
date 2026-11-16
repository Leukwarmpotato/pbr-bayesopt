import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class History:
    points: np.ndarray
    values: np.ndarray

    @property
    def best_so_far(self):
        return np.maximum.accumulate(self.values)

    @property
    def best_point(self):
        return self.points[np.argmax(self.values)]

    @property
    def best_value(self):
        return float(np.max(self.values))


def to_unit(points, lower, upper):
    return (np.asarray(points) - lower) / (upper - lower)


def from_unit(points, lower, upper):
    return lower + np.asarray(points) * (upper - lower)


def sobol_points(n, dimension, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d=dimension, scramble=True, seed=seed).random(n)


def fit_surrogate(unit_points, values, seed):
    dimension = unit_points.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.full(dimension, 0.3), length_scale_bounds=(0.03, 3.0), nu=2.5)
        + WhiteKernel(1e-6, (1e-8, 1e-2))
    )
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(unit_points, values)
    return model


def expected_improvement(model, candidates, best_value, exploration=0.01):
    mean, std = model.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-12)
    improvement = mean - best_value - exploration * np.abs(best_value)
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def propose(model, unit_points, values, dimension, seed, n_candidates=4096):
    candidates = sobol_points(n_candidates, dimension, seed)
    scores = expected_improvement(model, candidates, np.max(values))
    distances = np.min(np.linalg.norm(candidates[:, None, :] - unit_points[None, :, :], axis=2), axis=1)
    scores[distances < 1e-3] = -np.inf
    return candidates[np.argmax(scores)]


def bayesian_optimisation(objective, lower, upper, n_initial=8, n_iterations=22, seed=0, callback=None):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    dimension = len(lower)

    unit_points = sobol_points(n_initial, dimension, seed)
    values = np.array([objective(from_unit(point, lower, upper)) for point in unit_points])

    for iteration in range(n_iterations):
        model = fit_surrogate(unit_points, values, seed)
        candidate = propose(model, unit_points, values, dimension, seed=1000 * (seed + 1) + iteration)
        value = objective(from_unit(candidate, lower, upper))
        unit_points = np.vstack([unit_points, candidate])
        values = np.append(values, value)
        if callback is not None:
            callback(iteration, from_unit(candidate, lower, upper), value, model)

    return History(points=from_unit(unit_points, lower, upper), values=values)


def random_search(objective, lower, upper, n_evaluations=30, seed=0):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    points = from_unit(rng.random((n_evaluations, len(lower))), lower, upper)
    values = np.array([objective(point) for point in points])
    return History(points=points, values=values)
