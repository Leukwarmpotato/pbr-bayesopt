import numpy as np

from pbr_bayesopt.optimiser import bayesian_optimisation, expected_improvement, fit_surrogate, random_search

LOWER = np.array([-2.0, -1.0])
UPPER = np.array([3.0, 4.0])
PEAK = np.array([1.0, 2.0])


def smooth_peak(point):
    return float(np.exp(-np.sum((np.asarray(point) - PEAK) ** 2)))


def test_finds_peak_of_smooth_function():
    history = bayesian_optimisation(smooth_peak, LOWER, UPPER, n_initial=8, n_iterations=17, seed=0)
    assert history.best_value > 0.97
    assert np.linalg.norm(history.best_point - PEAK) < 0.2


def test_beats_random_search_on_average():
    bo = [bayesian_optimisation(smooth_peak, LOWER, UPPER, n_initial=8, n_iterations=12, seed=s).best_value for s in range(3)]
    rs = [random_search(smooth_peak, LOWER, UPPER, n_evaluations=20, seed=s).best_value for s in range(3)]
    assert np.mean(bo) > np.mean(rs)


def test_same_seed_gives_same_run():
    first = bayesian_optimisation(smooth_peak, LOWER, UPPER, n_initial=4, n_iterations=3, seed=1)
    second = bayesian_optimisation(smooth_peak, LOWER, UPPER, n_initial=4, n_iterations=3, seed=1)
    assert np.allclose(first.points, second.points)


def test_points_stay_inside_bounds():
    history = bayesian_optimisation(smooth_peak, LOWER, UPPER, n_initial=4, n_iterations=4, seed=2)
    assert np.all(history.points >= LOWER) and np.all(history.points <= UPPER)


def test_expected_improvement_is_non_negative():
    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    values = np.array([smooth_peak(LOWER + p * (UPPER - LOWER)) for p in points])
    model = fit_surrogate(points, values, seed=0)
    scores = expected_improvement(model, rng.random((200, 2)), values.max())
    assert np.all(scores >= 0)


def test_best_so_far_never_decreases():
    history = random_search(smooth_peak, LOWER, UPPER, n_evaluations=15, seed=0)
    assert np.all(np.diff(history.best_so_far) >= 0)
