from dataclasses import dataclass, field

import numpy as np

from pbr_bayesopt.reactor import ReactorParams, solve_steady_state

DEFAULT_BOUNDS = {"light": (100.0, 1000.0), "dilution_rate": (0.05, 0.6)}
TARGETS = {
    "energy_yield": "Energy yield (g biomass per kWh)",
    "productivity": "Volumetric productivity (g m⁻³ day⁻¹)",
}


@dataclass
class ReactorObjective:
    target: str = "energy_yield"
    bounds: dict = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    params: ReactorParams = field(default_factory=ReactorParams)

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}")

    @property
    def names(self):
        return list(self.bounds)

    @property
    def lower(self):
        return np.array([self.bounds[name][0] for name in self.names])

    @property
    def upper(self):
        return np.array([self.bounds[name][1] for name in self.names])

    @property
    def label(self):
        return TARGETS[self.target]

    def __call__(self, point):
        light, dilution_rate = np.clip(point, self.lower, self.upper)
        state = solve_steady_state(light, dilution_rate, self.params)
        return getattr(state, self.target)
