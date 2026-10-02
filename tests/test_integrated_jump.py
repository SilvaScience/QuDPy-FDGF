"""Integrated jump observables: exact Heisenberg-picture window integral.

The default ``integration="exact"`` propagates the detector gamma L^dagger L once
in the Heisenberg picture and integrates it over the window; the legacy
``integration="trapezoid"`` samples the window. Checks: analytical values for an
open two-level system (window starting at 0 or later), agreement with a fine
quadrature for a coherent, dissipative multi-level model, dense/sparse agreement,
efficiency scaling, and input validation.
"""
import numpy as np
import pytest

from qudpy_fdgf import (
    EigenbasisKModel,
    ExcitationSectorModel,
    FrequencyPathway,
    ObservableSpec,
    SpectroscopySolver,
)
from qudpy_fdgf.protocols import standard_nq_protocol

GAMMA_1, GAMMA_PHI = 0.04, 0.03
H = np.diag([0.0, 2.0]).astype(complex)
MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
P_E = np.diag([0.0, 1.0]).astype(complex)
L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([-1.0, 1.0]).astype(complex)
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1,
                                detection_interval=3, nq_axis="w1", detection_axis="w3")
R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing")
REPHASING = (
    FrequencyPathway(name="GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="SE", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
BACKENDS = ("dense", "sparse_sector")


def two_level_solver(backend):
    solver = SpectroscopySolver(backend=backend, eta=0.002)
    solver.feed_model(EigenbasisKModel(
        H, MU, c_ops_raw=((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)),
        observable_op_arrays={"polarization": MU, "excited_population": P_E}))
    return solver


def three_site_chain(dissipative=True):
    """Hard-core exciton chain, manifolds 0-2, local dephasing and collective decay."""
    n, J = 3, -0.05
    pairs = [(0, 1), (0, 2), (1, 2)]
    eps = 2.0 + 0.01 * np.arange(n)
    H1 = np.diag(eps).astype(complex) + J * (np.eye(n, k=1) + np.eye(n, k=-1))
    H2 = np.diag([eps[i] + eps[j] for i, j in pairs]).astype(complex)
    H2[0, 1] = H2[1, 0] = J          # (0,1) <-> (0,2)
    H2[1, 2] = H2[2, 1] = J          # (0,2) <-> (1,2)
    up10 = np.ones((n, 1), dtype=complex)
    up21 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=complex)
    number = [{("1", "1"): np.diag(np.eye(n)[s]).astype(complex),
               ("2", "2"): np.diag([1.0 if s in p else 0.0 for p in pairs]).astype(complex)}
              for s in range(n)]
    lowering = {("0", "1"): up10.conj().T, ("1", "2"): up21.conj().T}
    c_ops = tuple((blocks, 0.02) for blocks in number) + ((lowering, 0.005),)
    return ExcitationSectorModel({"0": np.zeros((1, 1), dtype=complex), "1": H1, "2": H2},
                                 {("1", "0"): up10, ("2", "1"): up21},
                                 c_ops_raw=c_ops if dissipative else (),
                                 initial_sector="0")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("window", [(0.0, 125.0), (10.0, 60.0)])
def test_two_level_window_integral_is_exact(backend, window):
    solver = two_level_solver(backend)
    axes = {"w1": np.linspace(-2.5, -1.5, 3), "w3": np.linspace(1.5, 2.5, 3)}
    observables = {
        "pop": ObservableSpec.action("pop", fourth_interaction="Bu", operator="excited_population"),
        "fl": ObservableSpec.mean_jump("fl", "c_op_0", time_window=window, fourth_interaction="Bu"),
    }
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=(R1,),
                                      fixed_coordinates={"t2": 10.0}, observables=observables)
    t0, t1 = window
    expected = np.exp(-GAMMA_1 * t0) - np.exp(-GAMMA_1 * t1)
    ratio = result.observables["fl"]["R1"] / result.observables["pop"]["R1"]
    assert np.max(np.abs(ratio - expected)) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_efficiency_scales_the_count(backend):
    solver = two_level_solver(backend)
    axes = {"w1": np.array([-2.0]), "w3": np.array([2.0])}
    spec = dict(time_window=(0.0, 50.0), fourth_interaction="Bu")
    observables = {"full": ObservableSpec.mean_jump("full", "c_op_0", **spec),
                   "part": ObservableSpec.mean_jump("part", "c_op_0", efficiency=0.35, **spec)}
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=(R1,),
                                      fixed_coordinates={"t2": 0.0}, observables=observables)
    full = result.observables["full"]["R1"][0, 0]
    assert result.observables["part"]["R1"][0, 0] == pytest.approx(0.35 * full, rel=1e-12)


def test_exact_matches_fine_quadrature_and_backends_agree():
    axes = {"w1": np.array([-2.0]), "w3": np.array([2.0])}       # one point: the quadrature is costly
    results = {}
    for backend in BACKENDS:
        solver = SpectroscopySolver(backend=backend, eta=0.005)
        solver.feed_model(three_site_chain())
        channels = solver.jump_channel_names()
        observables = {}
        for name in (channels[0], channels[-1]):          # local dephasing, collective decay
            window = dict(time_window=(5.0, 80.0), fourth_interaction="Bu")
            observables[f"exact_{name}"] = ObservableSpec.mean_jump(f"exact_{name}", name, **window)
            if backend == "dense":
                observables[f"trap_{name}"] = ObservableSpec.mean_jump(
                    f"trap_{name}", name, integration="trapezoid", n_steps=2001, **window)
        results[backend] = solver.generate_spectrum(
            PROTOCOL, axes, pathways=REPHASING, fixed_coordinates={"t2": 10.0},
            observables=observables).observable_components
    for name in (channels[0], channels[-1]):
        dense = results["dense"][f"exact_{name}"]["rephasing"]
        scale = np.max(np.abs(dense))
        sparse = results["sparse_sector"][f"exact_{name}"]["rephasing"]
        trapezoid = results["dense"][f"trap_{name}"]["rephasing"]
        assert np.max(np.abs(dense - sparse)) < 1e-10 * scale
        assert np.max(np.abs(dense - trapezoid)) < 1e-6 * scale     # quadrature error ~ dt^2


def test_integration_option_is_validated():
    with pytest.raises(ValueError):
        ObservableSpec.mean_jump("x", "c_op_0", time_window=(0.0, 1.0), integration="simpson")
    spec = ObservableSpec.mean_jump("x", "c_op_0", time_window=(0.0, 1.0), integration="Trapezoid")
    assert spec.integration == "trapezoid"
    assert ObservableSpec.mean_jump("y", "c_op_0", time_window=(0.0, 1.0)).integration == "exact"
