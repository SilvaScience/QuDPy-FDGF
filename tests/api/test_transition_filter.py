"""Automatic split of the interaction operator: what is kept, what is excluded, and the window."""
import warnings

import numpy as np
import pytest

from helpers import H, MU
from qudpy_fdgf import (
    EigenbasisKModel,
    ExcitationSectorModel,
    FrequencyPathway,
    ModelConsistencyWarning,
    PropagationInterval,
    SpectroscopyProtocol,
    SpectroscopySolver,
)

LINEAR = FrequencyPathway("lin", interactions=["Ku"], component="linear")
PROTOCOL = SpectroscopyProtocol(intervals=[PropagationInterval("omega", "frequency")])

# ladder g (0) - e (1.0) - f (1.05): the e-f transition lies far below the optical one
LADDER_H = np.diag([0.0, 1.0, 1.05]).astype(complex)
LADDER_MU = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]], dtype=complex)


def test_two_level_model_keeps_everything_and_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = EigenbasisKModel(H, MU)
    summary = model.transition_summary()
    assert summary["decomposition"] == "automatic_energy"
    assert summary["kept_fraction"] == pytest.approx(1.0)
    assert summary["degenerate_fraction"] == 0.0
    assert summary["outside_window_fraction"] == 0.0


def test_window_excludes_the_low_frequency_transition_and_reports_it():
    model = EigenbasisKModel(LADDER_H, LADDER_MU, transition_window=(0.5, 2.0))
    summary = model.transition_summary()
    # sum |mu_ij|^2 = 2 (0.25 + 0.09); the e-f pair (0.09, twice) is outside the window
    assert summary["outside_window_fraction"] == pytest.approx(0.09 / 0.34)
    assert summary["kept_fraction"] == pytest.approx(0.25 / 0.34)
    assert summary["transition_window"] == (0.5, 2.0)
    plus = model.transition_blocks("light_matter", "plus", "k0")["k0"]
    assert plus[1, 0] != 0 and plus[2, 1] == 0 and plus[2, 0] == 0


def test_without_window_the_low_frequency_transition_is_a_resonant_one():
    plus = EigenbasisKModel(LADDER_H, LADDER_MU).transition_blocks("light_matter", "plus", "k0")["k0"]
    assert plus[2, 1] == pytest.approx(0.3)           # driven like the optical transition


def test_window_changes_the_spectrum():
    values = {}
    for window in (None, (1.5, 3.0)):
        solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
        solver.feed_model(EigenbasisKModel(LADDER_H, LADDER_MU, transition_window=window))
        values[window] = solver.generate_spectrum(
            PROTOCOL, {"omega": np.array([1.0])}, pathways=[LINEAR]).pathways["lin"][0]
    assert abs(values[None]) > 1.0
    assert values[(1.5, 3.0)] == 0                    # no transition left in the window


def test_significant_degenerate_pair_warns_and_is_reported():
    energies = np.diag([0.0, 0.0, 1.0]).astype(complex)       # two degenerate ground states
    dipole = np.array([[0, 0.4, 0.5], [0.4, 0, 0], [0.5, 0, 0]], dtype=complex)
    with pytest.warns(ModelConsistencyWarning, match="closer than rwa_tol"):
        model = EigenbasisKModel(energies, dipole)
    summary = model.transition_summary()
    assert summary["n_degenerate_states"] == 2
    assert summary["degenerate_fraction"] == pytest.approx(0.32 / 0.82)


def test_degenerate_weight_does_not_depend_on_the_basis_of_the_degenerate_subspace():
    energies = np.diag([0.0, 0.0, 1.0]).astype(complex)
    dipole = np.array([[0, 0.4, 0.5], [0.4, 0, 0], [0.5, 0, 0]], dtype=complex)
    angle = 0.7
    rotation = np.eye(3, dtype=complex)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = rotation.conj().T @ dipole @ rotation
    with pytest.warns(ModelConsistencyWarning):
        fraction = EigenbasisKModel(energies, rotated).transition_summary()["degenerate_fraction"]
    assert fraction == pytest.approx(0.32 / 0.82)


def test_negligible_degenerate_element_does_not_warn():
    energies = np.diag([0.0, 0.0, 1.0]).astype(complex)
    dipole = np.array([[0, 1e-9, 0.5], [1e-9, 0, 0], [0.5, 0, 0]], dtype=complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EigenbasisKModel(energies, dipole)


def test_explicit_decomposition_has_no_filter_and_rejects_a_window():
    plus = np.array([[0, 0], [1, 0]], dtype=complex)
    model = EigenbasisKModel(H, MU, j_plus_array=plus, j_minus_array=plus.conj().T)
    assert model.transition_summary() == {"decomposition": "explicit"}
    with pytest.raises(ValueError, match="transition_window"):
        EigenbasisKModel(H, MU, j_plus_array=plus, j_minus_array=plus.conj().T, transition_window=(0.5, 2.0))


@pytest.mark.parametrize("window", [(2.0, 1.0), (-1.0, 2.0), (1.0, 1.0), (1.0,), "ab"])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        EigenbasisKModel(H, MU, transition_window=window)


def test_summary_of_the_solver_reports_the_split():
    solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
    solver.feed_model(EigenbasisKModel(LADDER_H, LADDER_MU, transition_window=(0.5, 2.0)))
    assert solver.summary()["transition_summary"]["outside_window_fraction"] == pytest.approx(0.09 / 0.34)
    sectors = ExcitationSectorModel({0: np.zeros((1, 1)), 1: np.array([[1.0]])}, {(1, 0): np.array([[1.0]])})
    solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
    solver.feed_model(sectors)
    assert solver.summary()["transition_summary"] == {"decomposition": "explicit_sector"}
