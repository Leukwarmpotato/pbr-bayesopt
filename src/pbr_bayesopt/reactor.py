from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

SECONDS_PER_DAY = 86400.0
N_SPECIES = 6


@dataclass(frozen=True)
class ReactorParams:
    velocity: float = 0.2
    tube_radius: float = 0.0375
    tube_length: float = 600.0
    n_nodes: int = 100
    total_volume: float = 5.0

    light: float = 600.0
    fibre_spacing: float = 0.05
    light_distribution_eff: float = 0.9
    wall_plug_eff: float = 0.5
    light_energy_factor: float = 0.217

    absorption: float = 133.33
    k_light: float = 384.9
    k_inhibition: float = 1707.0
    mu_max: float = 5.6328e-5
    arrhenius_a: float = 22.6350
    arrhenius_b: float = 17.4630
    energy_a: float = 1.0026
    energy_b: float = 0.5572
    gas_constant: float = 0.008314
    temperature: float = 298.15

    k_carbon: float = 0.26
    k_oxygen: float = 0.023
    k_nitrate: float = 0.01023
    k_phosphate: float = 0.03
    death_rate: float = 4.1667e-5

    kla_co2: float = 0.05
    kla_o2: float = 1.2e-2
    co2_saturation: float = 0.6
    o2_saturation: float = 0.008

    yield_co2: float = 41.7
    yield_o2: float = 1.4
    yield_nitrogen: float = 157.0
    yield_phosphorus: float = 1000.0
    molar_mass_n: float = 0.014
    molar_mass_p: float = 0.031

    k1: float = 4.45e-7
    k2: float = 4.69e-11
    kw: float = 1e-14

    feed_rate: float = 9.75e-6
    carbon_feed: float = 2.972
    nitrate_feed: float = 0.7
    phosphate_feed: float = 0.05
    oxygen_feed: float = 0.008
    alkalinity_feed: float = 2.9e-3
    biomass_initial: float = 0.05

    @property
    def tube_area(self):
        return np.pi * self.tube_radius**2

    @property
    def flow_rate(self):
        return self.velocity * self.tube_area

    @property
    def tube_volume(self):
        return self.tube_area * self.tube_length

    @property
    def tank_volume(self):
        return self.total_volume - self.tube_volume

    @property
    def node_length(self):
        return self.tube_length / self.n_nodes

    @property
    def lit_surface(self):
        return 2 * np.pi * self.tube_radius * self.tube_length

    @property
    def dilution_rate(self):
        return self.feed_rate * SECONDS_PER_DAY / self.total_volume

    def with_operating_point(self, light, dilution_rate):
        feed_rate = dilution_rate * self.total_volume / SECONDS_PER_DAY
        return replace(self, light=float(light), feed_rate=float(feed_rate))


@dataclass(frozen=True)
class SteadyState:
    light: float
    dilution_rate: float
    biomass: float
    productivity: float
    electrical_power: float
    energy_yield: float
    converged: bool


def temperature_factor(p):
    rt = p.gas_constant * p.temperature
    return p.arrhenius_a * np.exp(-p.energy_a / rt) - p.arrhenius_b * np.exp(-p.energy_b / rt)


def average_light(biomass, p):
    optical_depth = p.absorption * biomass * p.fibre_spacing
    incident = p.light_distribution_eff * p.light
    safe_depth = np.maximum(optical_depth, 1e-12)
    return np.where(optical_depth < 1e-12, incident, incident * (1 - np.exp(-safe_depth)) / safe_depth)


def growth_rate(biomass, carbon, nitrate, phosphate, oxygen, p):
    light = average_light(biomass, p)
    return (
        p.mu_max
        * temperature_factor(p)
        * (light / (p.k_light + light + light**2 / p.k_inhibition))
        * (carbon / (carbon + p.k_carbon))
        * (nitrate / (nitrate + p.k_nitrate))
        * (phosphate / (phosphate + p.k_phosphate))
        / (1.0 + oxygen / p.k_oxygen)
    )


def hydrogen_concentration(carbon_molar, alkalinity, p):
    h = np.clip(p.k1 * max(carbon_molar - alkalinity, carbon_molar * 0.01) / max(alkalinity, 1e-15), 1e-14, 1e-2)
    for _ in range(50):
        denominator = h**2 + p.k1 * h + p.k1 * p.k2
        bicarbonate = carbon_molar * p.k1 * h / denominator
        carbonate = carbon_molar * p.k1 * p.k2 / denominator
        residual = bicarbonate + 2 * carbonate + p.kw / h - h - alkalinity
        slope = (
            carbon_molar * p.k1 * (p.k1 * p.k2 - h**2) / denominator**2
            - 2 * carbon_molar * p.k1 * p.k2 * (2 * h + p.k1) / denominator**2
            - p.kw / h**2
            - 1
        )
        h = max(h - residual / slope, 1e-14)
    return h


def dissolved_co2(carbon, alkalinity, p):
    carbon_molar = carbon / 1000.0
    h = hydrogen_concentration(carbon_molar, alkalinity, p)
    return carbon_molar / (1 + p.k1 / h + p.k1 * p.k2 / h**2) * 1000.0


def derivatives(t, state, p):
    tank = state[:N_SPECIES]
    tube = state[N_SPECIES:].reshape(p.n_nodes, N_SPECIES)
    outlet = tube[-1]
    recycle = p.flow_rate / p.tank_volume
    feed = p.feed_rate / p.tank_volume

    d_tank = recycle * (outlet - tank)
    d_tank[0] -= feed * tank[0]
    d_tank[1] += feed * (p.carbon_feed - tank[1]) + p.kla_co2 * (p.co2_saturation - dissolved_co2(tank[1], tank[5], p))
    d_tank[2] += feed * (p.nitrate_feed - tank[2])
    d_tank[3] += feed * (p.phosphate_feed - tank[3])
    d_tank[4] += feed * (p.oxygen_feed - tank[4]) - p.kla_o2 * (tank[4] - p.o2_saturation)
    d_tank[5] += feed * (p.alkalinity_feed - tank[5])

    upstream = np.vstack([tank, tube[:-1]])
    biomass, carbon, nitrate, phosphate, oxygen, _ = tube.T
    growth = growth_rate(biomass, carbon, nitrate, phosphate, oxygen, p) * biomass

    d_tube = -p.velocity * (tube - upstream) / p.node_length
    d_tube[:, 0] += growth - p.death_rate * biomass**2
    d_tube[:, 1] -= p.yield_co2 * growth
    d_tube[:, 2] -= growth / (p.yield_nitrogen * p.molar_mass_n)
    d_tube[:, 3] -= growth / (p.yield_phosphorus * p.molar_mass_p)
    d_tube[:, 4] += p.yield_o2 * growth

    return np.concatenate([d_tank, d_tube.ravel()])


def jacobian_sparsity(p):
    n = N_SPECIES * (p.n_nodes + 1)
    pattern = lil_matrix((n, n), dtype=int)
    for node in range(p.n_nodes + 1):
        block = slice(node * N_SPECIES, (node + 1) * N_SPECIES)
        pattern[block, block] = 1
        if node > 0:
            pattern[block, (node - 1) * N_SPECIES:node * N_SPECIES] = 1
    pattern[:N_SPECIES, p.n_nodes * N_SPECIES:] = 1
    return pattern.tocsr()


def initial_state(p):
    node = np.array([p.biomass_initial, p.carbon_feed, p.nitrate_feed, p.phosphate_feed, p.oxygen_feed, p.alkalinity_feed])
    return np.tile(node, p.n_nodes + 1)


def electrical_power(p):
    return p.light * p.lit_surface * p.light_energy_factor / 1000.0 / p.wall_plug_eff


def solve_steady_state(light, dilution_rate, params=None, max_days=350.0, tolerance=1e-9):
    p = (params or ReactorParams()).with_operating_point(light, dilution_rate)

    def settled(t, state, p):
        rate = np.abs(derivatives(t, state, p))
        return np.max(rate / (np.abs(state) + 1e-6)) - tolerance

    settled.terminal = True
    settled.direction = -1

    solution = solve_ivp(
        derivatives,
        (0.0, max_days * SECONDS_PER_DAY),
        initial_state(p),
        args=(p,),
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        jac_sparsity=jacobian_sparsity(p),
        events=settled,
    )

    biomass = max(0.0, float(solution.y[0, -1]))
    productivity = p.dilution_rate * biomass * 1000.0
    power = electrical_power(p)
    daily_energy = power * 24.0
    energy_yield = productivity * p.total_volume / daily_energy
    return SteadyState(
        light=float(light),
        dilution_rate=float(dilution_rate),
        biomass=biomass,
        productivity=productivity,
        electrical_power=power,
        energy_yield=energy_yield,
        converged=bool(solution.status == 1),
    )
