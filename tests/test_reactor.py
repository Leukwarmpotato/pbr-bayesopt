import pytest

from pbr_bayesopt.reactor import ReactorParams, electrical_power, solve_steady_state


def test_design_point_matches_julia_model():
    state = solve_steady_state(light=600.0, dilution_rate=0.16848)
    assert state.converged
    assert state.biomass == pytest.approx(0.143, abs=0.002)
    assert state.productivity == pytest.approx(24.1, abs=0.4)


def test_led_power_at_design_point():
    assert electrical_power(ReactorParams()) == pytest.approx(36.8, abs=0.1)


def test_low_light_and_fast_dilution_washes_out():
    state = solve_steady_state(light=100.0, dilution_rate=0.5)
    assert state.biomass < 1e-4
    assert state.energy_yield < 1e-3


def test_more_light_gives_more_biomass():
    dim = solve_steady_state(light=300.0, dilution_rate=0.15)
    bright = solve_steady_state(light=800.0, dilution_rate=0.15)
    assert bright.biomass > dim.biomass

