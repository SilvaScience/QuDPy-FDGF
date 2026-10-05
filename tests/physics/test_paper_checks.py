"""Fast analytical checks taken from the manuscript (Example 1 and the Quick start).

Each check compares a numerical result with a closed-form value:
- linear absorption of an open two-level system: FWHM = 2 (Gamma_2 + eta);
- third-order rephasing peak: |S| = 2 mu^4 / (Gamma_2 + eta)^2 at (-omega_eg, omega_eg), t2 = 0;
- complete rephasing response at t2 > 0: R1 + R2 + R3 = -i mu^4 2 exp(-gamma_1 t2) / [...], where R3 = (Bu, Ku, Ku)
  is non-zero only because the radiative channel refills the ground state (R1 + R2 alone give 1 + exp(-gamma_1 t2));
- action-detected population: S_pop = -i S_pol (Eq. action_pol_identity);
- integrated fluorescence: N_rad / P_e = 1 - exp(-gamma_1 T), exactly;
- dense and sparse backends agree for all three observables of Example 1.

The same relations are shown, with figures, in validation/analysis/example1_analysis.ipynb.
"""
import numpy as np
import pytest

from qudpy_fdgf import (
    EigenbasisKModel,
    FrequencyPathway,
    ObservableSpec,
    SpectroscopySolver,
)
from qudpy_fdgf.protocols import (
    PropagationInterval,
    SpectroscopyProtocol,
    standard_nq_protocol,
)

H = np.diag([0.0, 2.0]).astype(complex)
MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
P_E = np.diag([0.0, 1.0]).astype(complex)
L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([-1.0, 1.0]).astype(complex)
GAMMA_1, GAMMA_PHI = 0.04, 0.03
GAMMA_2 = GAMMA_1 / 2 + GAMMA_PHI

R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"),
                      component="rephasing", detection="polarization")
R2 = FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"),
                      component="rephasing", detection="polarization")
R3 = FrequencyPathway(name="R3", interactions=("Bu", "Ku", "Ku"),
                      component="rephasing", detection="polarization")
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1,
                                detection_interval=3, nq_axis="omega_1q",
                                detection_axis="omega_emit")


def example1_solver(backend="dense", eta=0.002, gamma_1=GAMMA_1):
    channels = ((L_RAD, gamma_1), (SIGMA_Z, GAMMA_PHI / 2)) if gamma_1 > 0 else ((SIGMA_Z, GAMMA_PHI / 2),)
    model = EigenbasisKModel(
        H, MU,
        c_ops_raw=channels,
        observable_op_arrays={"polarization": MU, "excited_population": P_E},
    )
    solver = SpectroscopySolver(backend=backend, eta=eta)
    solver.feed_model(model)
    return solver


def example1_observables(solver, n_steps=101):
    return {
        "polarization": "polarization",
        "population": ObservableSpec.action(
            "population", fourth_interaction="Bu", operator="excited_population"),
        "fluorescence": ObservableSpec.mean_jump(
            "fluorescence", solver.jump_channel_names()[0],
            time_window=(0.0, 5 / GAMMA_1), n_steps=n_steps, fourth_interaction="Bu"),
    }


def test_linear_width_matches_coherence_decay():
    solver = example1_solver(eta=1e-4)
    linear = FrequencyPathway("linear", interactions=[{"label": "Ku"}], component="linear")
    protocol = SpectroscopyProtocol(intervals=[PropagationInterval("omega1", "frequency")])
    w = np.linspace(1.8, 2.2, 8001)
    signal = solver.generate_spectrum(protocol, {"omega1": w}, pathways=[linear]).pathways["linear"]
    absorptive = signal.imag
    half = w[absorptive >= absorptive.max() / 2]
    assert w[np.argmax(absorptive)] == pytest.approx(2.0, abs=1e-4)
    assert half[-1] - half[0] == pytest.approx(2 * (GAMMA_2 + 1e-4), abs=2e-4)


def test_rephasing_peak_amplitude_and_position():
    eta = 0.002
    solver = example1_solver(eta=eta)
    axes = {"omega_1q": np.array([-2.0]), "omega_emit": np.array([2.0])}
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=(R1, R2),
                                      fixed_coordinates={"t2": 0.0})
    peak = abs(result.components["rephasing"][0, 0])
    assert peak == pytest.approx(2 * 0.5**4 / (GAMMA_2 + eta) ** 2, rel=1e-10)


def test_complete_rephasing_response_includes_the_relaxation_pathway():
    eta, t2, d = 0.002, 10.0, 0.5
    solver = example1_solver(eta=eta)
    w1 = np.linspace(-2.3, -1.7, 7)
    w3 = np.linspace(1.7, 2.3, 7)
    result = solver.generate_spectrum(PROTOCOL, {"omega_1q": w1, "omega_emit": w3}, pathways=(R1, R2, R3),
                                      fixed_coordinates={"t2": t2})
    total = sum(result.pathways[name] for name in ("R1", "R2", "R3"))
    two = result.pathways["R1"] + result.pathways["R2"]
    a, b = np.meshgrid(w1, w3, indexing="ij")
    lineshape = 1.0 / ((GAMMA_2 + eta - 1j * (a + 2.0)) * (GAMMA_2 + eta - 1j * (b - 2.0)))
    complete = -1j * d**4 * 2 * np.exp(-GAMMA_1 * t2) * lineshape
    without_r3 = -1j * d**4 * (1 + np.exp(-GAMMA_1 * t2)) * lineshape
    assert np.max(np.abs(total - complete)) <= 1e-12 * np.max(np.abs(complete))
    assert np.max(np.abs(two - without_r3)) <= 1e-12 * np.max(np.abs(without_r3))
    # R3 has the sign opposite to R2 and the weight 1 - exp(-gamma_1 t2)
    ratio = result.pathways["R3"] / result.pathways["R2"]
    assert np.max(np.abs(ratio + (1 - np.exp(-GAMMA_1 * t2)))) <= 1e-12


def test_relaxation_pathway_vanishes_without_radiative_decay():
    solver = example1_solver(gamma_1=0.0)
    axes = {"omega_1q": np.array([-2.0]), "omega_emit": np.array([2.0])}
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=(R1, R2, R3), fixed_coordinates={"t2": 10.0})
    assert abs(result.pathways["R3"][0, 0]) <= 1e-12 * abs(result.pathways["R2"][0, 0])
    assert result.pathways["R1"][0, 0] == pytest.approx(result.pathways["R2"][0, 0], rel=1e-12)


def test_action_population_and_fluorescence():
    solver = example1_solver()
    axes = {"omega_1q": np.linspace(-2.5, -1.5, 5), "omega_emit": np.linspace(1.5, 2.5, 5)}
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=(R1, R2, R3),
                                      fixed_coordinates={"t2": 10.0},
                                      observables=example1_observables(solver))
    for name in ("R1", "R2", "R3"):
        s_pol = result.observables["polarization"][name]
        s_pop = result.observables["population"][name]
        assert np.max(np.abs(s_pop + 1j * s_pol)) <= 1e-12 * np.max(np.abs(s_pol))
    ratio = result.observables["fluorescence"]["R1"] / result.observables["population"]["R1"]
    exact = 1 - np.exp(-5.0)
    assert np.max(np.abs(ratio - exact)) / exact < 1e-12          # exact window integral


def test_dense_and_sparse_backends_agree_for_example1():
    axes = {"omega_1q": np.linspace(-2.5, -1.5, 3), "omega_emit": np.linspace(1.5, 2.5, 3)}
    results = {}
    for backend in ("dense", "sparse_sector"):
        solver = example1_solver(backend)
        results[backend] = solver.generate_spectrum(
            PROTOCOL, axes, pathways=(R1, R2, R3), fixed_coordinates={"t2": 10.0},
            observables=example1_observables(solver))
    for name in ("polarization", "population", "fluorescence"):
        for pathway in ("R1", "R2", "R3"):
            a = results["dense"].observables[name][pathway]
            b = results["sparse_sector"].observables[name][pathway]
            assert np.max(np.abs(a - b)) <= 1e-10 * np.max(np.abs(a))
