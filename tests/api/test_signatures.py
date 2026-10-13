"""Signatures of pathways, validation of components, enumeration and the null-pathway report."""
import warnings

import numpy as np
import pytest

from helpers import GAMMA_1, GAMMA_PHI, H, L_RAD, MU, PROTOCOL, SIGMA_Z
from qudpy_fdgf import (
    EigenbasisKModel,
    FrequencyPathway,
    Interaction,
    NullPathwayWarning,
    PropagationInterval,
    SpectroscopyProtocol,
    SpectroscopySolver,
    pathway_signature,
    pathways_from_signature,
    translate_ufss_diagrams,
)

RELAXING = ((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2))


def two_level_solver(channels=RELAXING, backend="dense"):
    solver = SpectroscopySolver(backend=backend, eta=0.002)
    solver.feed_model(EigenbasisKModel(H, MU, c_ops_raw=channels))
    return solver


def test_signs_of_the_four_labels_and_of_standard_pathways():
    assert [Interaction(label).sign for label in ("Ku", "Kd", "Bu", "Bd")] == [1, -1, -1, 1]
    assert pathway_signature(("Bu", "Ku", "Bd")) == (-1, 1, 1)                    # rephasing
    assert pathway_signature([Interaction("Ku"), Interaction("Bu"), Interaction("Bd")]) == (1, -1, 1)
    assert FrequencyPathway("R", interactions=("Bu", "Bd", "Ku"), component="rephasing").signature == (-1, 1, 1)
    assert pathway_signature(("Bu", "Bu", "Ku", "Ku", "Bd")) == (-1, -1, 1, 1, 1)


def test_a_known_component_rejects_the_wrong_signature():
    with pytest.raises(ValueError, match="signature"):
        FrequencyPathway("bad", interactions=("Ku", "Bu", "Bd"), component="rephasing")
    with pytest.raises(ValueError, match="signature"):
        FrequencyPathway("bad", interactions=("Bu", "Ku", "Bd"), component="nonrephasing")
    FrequencyPathway("ok", interactions=("Ku", "Bu", "Bd"), component="nonrephasing")
    FrequencyPathway("ok", interactions=("Ku", "Bu", "Bd"), component="unrephasing")


def test_other_components_and_other_orders_are_not_constrained():
    FrequencyPathway("mixed", interactions=("Ku", "Bu", "Bd"), component="custom")
    FrequencyPathway("also", interactions=("Bu", "Ku", "Bd"), component="custom")
    FrequencyPathway("fourth", interactions=("Bu", "Ku", "Bd", "Bu"), component="rephasing")   # four interactions


def test_enumeration_returns_all_sequences_with_the_signature():
    pathways = pathways_from_signature("-++", component="rephasing")
    assert len(pathways) == 8
    assert {p.signature for p in pathways} == {(-1, 1, 1)}
    assert len({p.name for p in pathways}) == 8
    assert {p.origin for p in pathways} == {"enumerated"}
    assert {p.component for p in pathways} == {"rephasing"}
    assert "Bu-Ku-Bd" in {p.name for p in pathways}
    assert len(pathways_from_signature([-1, -1, 1, 1, 1])) == 32


def test_enumeration_accepts_several_notations_and_rejects_nonsense():
    base = [p.name for p in pathways_from_signature((-1, 1, 1))]
    assert [p.name for p in pathways_from_signature("-++")] == base
    assert [p.name for p in pathways_from_signature("(-, +, +)")] == base
    assert [p.name for p in pathways_from_signature("−++")] == base
    for bad in ("", "-+x", [0, 1], [2, 1]):
        with pytest.raises(ValueError):
            pathways_from_signature(bad)


def test_manifold_filter_reproduces_the_ufss_rule():
    ground_only = pathways_from_signature("-++", component="rephasing", max_manifold=1, min_manifold=0)
    assert {p.name for p in ground_only} == {"Bu-Ku-Bd", "Bu-Bd-Ku"}                      # R1 and R2
    two = pathways_from_signature("-++", component="rephasing", max_manifold=2, min_manifold=0)
    assert {p.name for p in two} == {"Bu-Ku-Bd", "Bu-Bd-Ku", "Bu-Ku-Ku"}              # adds R3


def test_ufss_translation_marks_its_pathways():
    pathways = translate_ufss_diagrams([[("Bu", 0), ("Ku", 1), ("Bd", 2)]], component="rephasing")
    assert pathways[0].origin == "ufss"
    assert pathways[0].metadata()["signature"] == (-1, 1, 1)


AXES = {"w1": np.array([-2.0]), "w3": np.array([2.0])}
FIXED = {"t2": 10.0}


def test_enumerated_set_gives_the_complete_response_of_the_open_two_level_system():
    solver = two_level_solver()
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED,
                                      pathways=pathways_from_signature("-++", component="rephasing"))
    total = result.components["rephasing"][0, 0]
    gamma_2, eta, d = GAMMA_1 / 2 + GAMMA_PHI, 0.002, 0.5
    expected = -2j * d**4 * np.exp(-GAMMA_1 * 10.0) / (gamma_2 + eta) ** 2        # 2 exp(-gamma_1 t2)
    assert total == pytest.approx(expected, rel=1e-10)
    # five of the eight sequences vanish, the three of the article do not
    assert set(result.null_pathways()) == {"Bu-Bd-Bd", "Kd-Ku-Ku", "Kd-Ku-Bd", "Kd-Bd-Ku", "Kd-Bd-Bd"}


def test_enumerated_zeros_do_not_warn_but_are_listed():
    solver = two_level_solver()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED,
                                          pathways=pathways_from_signature("-++", component="rephasing"))
    assert len(result.diagnostics["null_pathways"]) == 5


def test_a_hand_written_zero_pathway_warns_and_the_flag_silences_it():
    solver = two_level_solver(channels=((SIGMA_Z, GAMMA_PHI / 2),))               # no relaxation: R3 vanishes
    r3 = FrequencyPathway("R3", interactions=("Bu", "Ku", "Ku"), component="rephasing")
    r2 = FrequencyPathway("R2", interactions=("Bu", "Bd", "Ku"), component="rephasing")
    with pytest.warns(NullPathwayWarning, match="'R3'"):
        result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=(r2, r3))
    assert result.null_pathways() == ("R3",)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=(r2, r3), warn_null=False)


def test_a_result_in_which_everything_vanishes_warns_whatever_the_origin():
    solver = two_level_solver()
    # (Ku, Ku, Ku) cannot be read by the polarization of a two-level system
    pathways = [FrequencyPathway("KKK", interactions=("Ku", "Ku", "Ku"), origin="enumerated")]
    with pytest.warns(NullPathwayWarning):
        solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=pathways)


def test_null_pathways_uses_the_largest_pathway_as_scale():
    solver = two_level_solver()
    result = solver.generate_spectrum(
        PROTOCOL, AXES, fixed_coordinates=FIXED, warn_null=False,
        pathways=[FrequencyPathway("R2", interactions=("Bu", "Bd", "Ku"), component="rephasing")])
    assert result.null_pathways() == ()
    assert result.null_pathways(tolerance=2.0) == ("R2",)         # tolerance larger than 1 flags everything


def test_signed_interval_accepts_only_the_declared_sign():
    pathway = FrequencyPathway("R2", interactions=("Bu", "Bd", "Ku"), component="rephasing")   # q = (-1, 0, +1)
    solver = two_level_solver()
    for signed, coherence, ok in ((False, 1, True), (False, -1, True), (True, -1, True), (True, 1, False)):
        protocol = SpectroscopyProtocol(intervals=(
            PropagationInterval("w1", "frequency", coherence, signed=signed),
            PropagationInterval("t2", "time"),
            PropagationInterval("w3", "frequency"),
        ))
        if ok:
            solver.generate_spectrum(protocol, AXES, fixed_coordinates=FIXED, pathways=[pathway])
        else:
            with pytest.raises(ValueError, match="signed"):
                solver.generate_spectrum(protocol, AXES, fixed_coordinates=FIXED, pathways=[pathway])
