"""The Boltzmann constant has no default: a thermal state needs the unit of the Hamiltonian."""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, ExcitationSectorModel, SpectroscopySolver, ThermodynamicContext
from qudpy_fdgf.model_adapters import _BOLTZMANN_CONSTANTS

# the same gap, 50 meV, in each unit (CODATA 2018)
GAP = {"eV": 0.05, "meV": 50.0, "cm-1": 50.0 * 8.065543937, "THz": 0.05 * 241.7989242}
T = 100.0
K_B_EV = 8.617333262e-5


def gibbs_excited_population(gap_ev, temperature):
    weight = np.exp(-gap_ev / (K_B_EV * temperature))
    return weight / (1.0 + weight)


def eigenbasis_model(gap, **unit):
    return EigenbasisKModel(np.diag([0.0, gap]).astype(complex), np.array([[0, 1], [1, 0]], dtype=complex), **unit)


def sector_model(gap, **unit):
    return ExcitationSectorModel({0: np.zeros((1, 1)), 1: np.array([[gap]])}, {(1, 0): np.array([[1.0]])},
                                 initial_sector=0, **unit)


def excited_population(model, temperature=T):
    solver = SpectroscopySolver(backend="dense", eta=0.001, check_stationarity="off")
    solver.feed_model(model, context=ThermodynamicContext(temperature=temperature))
    return float(np.real(np.diag(solver.backend._initial_density_matrix)[-1]))


@pytest.mark.parametrize("unit", sorted(GAP))
@pytest.mark.parametrize("factory", (eigenbasis_model, sector_model))
def test_same_physical_gap_gives_the_same_gibbs_state_in_every_unit(factory, unit):
    expected = gibbs_excited_population(0.05, T)
    assert excited_population(factory(GAP[unit], energy_unit=unit)) == pytest.approx(expected, rel=1e-7)


def test_explicit_boltzmann_constant_is_the_same_as_the_unit():
    explicit = excited_population(eigenbasis_model(50.0, boltzmann_constant=_BOLTZMANN_CONSTANTS["meV"]))
    assert explicit == excited_population(eigenbasis_model(50.0, energy_unit="meV"))


def test_constants_are_k_b_over_the_unit_of_energy():
    assert _BOLTZMANN_CONSTANTS["eV"] == pytest.approx(8.617333262e-5, rel=1e-12)
    assert _BOLTZMANN_CONSTANTS["meV"] == pytest.approx(1e3 * _BOLTZMANN_CONSTANTS["eV"], rel=1e-12)
    assert _BOLTZMANN_CONSTANTS["cm-1"] == pytest.approx(0.6950348, rel=1e-7)           # k_B / (h c)
    assert _BOLTZMANN_CONSTANTS["THz"] == pytest.approx(0.02083661912, rel=1e-9)        # k_B / h, ordinary frequency


@pytest.mark.parametrize("factory", (eigenbasis_model, sector_model))
def test_a_temperature_without_a_unit_is_an_error_that_names_the_options(factory):
    solver = SpectroscopySolver(backend="dense", eta=0.001)
    with pytest.raises(ValueError, match="energy_unit") as error:
        solver.feed_model(factory(50.0), context=ThermodynamicContext(temperature=T))
    assert "boltzmann_constant" in str(error.value) and "no default" in str(error.value)


@pytest.mark.parametrize("factory", (eigenbasis_model, sector_model))
def test_zero_temperature_needs_no_unit(factory):
    assert excited_population(factory(50.0), temperature=0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("factory", (eigenbasis_model, sector_model))
def test_unit_and_constant_together_or_an_unknown_unit_are_errors(factory):
    with pytest.raises(ValueError, match="not both"):
        factory(50.0, energy_unit="meV", boltzmann_constant=0.0862)
    with pytest.raises(ValueError, match="Unknown energy_unit") as error:
        factory(50.0, energy_unit="kcal/mol")
    assert all(name in str(error.value) for name in ("eV", "meV", "cm-1", "THz"))
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="boltzmann_constant"):
            factory(50.0, boltzmann_constant=invalid)
