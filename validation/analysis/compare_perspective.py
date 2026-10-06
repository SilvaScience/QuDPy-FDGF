"""Compare QuDPy-FDGF with the stored data of the Perspective repository (Paiva-Ortega et al., 2DES, 2026).

The Perspective stores, for each scenario, the antidiagonal cuts of its rephasing spectra (`antidiagonal_data_dict.npy`):
ten samples of |S| along the line (omega_1, omega_3) = (-w, +w) on the frequency grid of its time-domain calculation,
for each Liouville diagram and for their sum, and the half width of the Lorentzian fitted to these ten points
(Table I of the Perspective). Nothing is recomputed with QuDPy: this script evaluates QuDPy-FDGF at the same ten points
and applies the same estimator.

    python compare_perspective.py "<Perspective repo>/Data/Borealis Data"

(or set the environment variable PERSPECTIVE_DATA). Results: ../results/data/perspective_comparison.json.

How the Perspective's numbers are produced (from its scripts, `Data/Borealis Data/Codes/AD+CE_only_rephasing.py`):

- model: two modes with three levels, H = sum_m [w_m n_m + 0.005 a_m^+ a_m^+ a_m a_m] + J (a^+ b + b^+ a); collapse operators
  sqrt(gamma) * L for dephasing (n_m), relaxation (a_m), transfer (|1_a><1_b| in local form) and, for the action signal only,
  radiative decay; start from the ground state; t2 = 10 eV^-1;
- time grid: N = int(801 * Reso) = 382 points on [0, 801] for t1 and t3 (step 801 / 381), Reso = 1.5 / pi, no apodization,
  `ifft2`; the frequency labels use the step 1 / Reso instead of the true step, so that the physical frequency of a label is
  label * (1 / Reso) / (801 / 381); the label step is 7.85 meV;
- coherent signal X3: R1 + R2 - R3 (UFSS rephasing diagrams, ESA subtracted);
- action signal X4: the four UFSS fluorescence diagrams Q3a (SE + Bu), Q4a (GSB + Bu), Q2b (ESA + Bu) and Q8b (ESA + Kd),
  detector N_exc after a delay of 5 eV^-1, added without signs (`np.sum`);
- linewidth: Lorentzian A g^2 / ((x - x0)^2 + g^2) fitted to the ten samples |S(-w, w)|, w in [0.96, 1.04] or [1.16, 1.24] eV.

Raw QuDPy diagrams carry no factor i per interaction, whereas the Liouville-space signals of QuDPy-FDGF do: for a diagram
with n_B bra-side and n_K ket-side interactions, S_FDGF = (+i)^n_B (-i)^n_K * S_raw. The signs of the raw diagrams are
therefore (-1, -1, +1, -1) for (Q3a, Q4a, Q2b, Q8b); the physical sum has the weights (+1, +1, +1, +1) on the FDGF signals.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # validation/ (models.py)
import models
from qudpy_fdgf import EigenbasisKModel, ObservableSpec, SpectroscopySolver

# ------------------------------------------------------------------------------------ the Perspective's grid
RESO = 1.5 / np.pi
N_POINTS = int(801 * RESO)                      # 382
DT_TRUE = 801.0 / (N_POINTS - 1)
DT_LABEL = 1.0 / RESO
LABEL_TO_TRUE = DT_LABEL / DT_TRUE              # physical frequency = label * LABEL_TO_TRUE
LABEL_STEP = 2 * np.pi * RESO / N_POINTS        # 7.85 meV
AMPLITUDE_SCALE = (N_POINTS * DT_TRUE) ** 2     # |S_continuous| = AMPLITUDE_SCALE * |ifft2 value|
ETA = 1e-5
T2 = 10.0

SCENARIOS = {                                   # folder of the Perspective repository -> change to EX4
    "Reference": {},
    "Enhance Transfer trans = 0.01": {"k_transfer": 0.010},
    "Trans =0.02": {"k_transfer": 0.020},
    "Trans =  1E-3": {"k_transfer": 1e-3},
    "Trans=1E-4": {"k_transfer": 1e-4},
    "k1 = 0.02": {"k_relax": 0.020},
}
RAW_SIGNS = (-1, -1, 1, -1)                     # S_FDGF = RAW_SIGNS * S_raw for (Q3a, Q4a, Q2b, Q8b)
DIAGRAMS = (("Q3a", "R_SE", "Bu"), ("Q4a", "R_GSB", "Bu"), ("Q2b", "R_ESA", "Bu"), ("Q8b", "R_ESA", "Kd"))


def load_scenario(root, folder):
    return np.load(Path(root) / folder / "antidiagonal_data_dict.npy", allow_pickle=True).item()


def lorentzian_halfwidth(labels, values):
    """The Perspective's estimator: A g^2 / ((x - x0)^2 + g^2) fitted to the samples; returns g in meV."""
    def model(x, amplitude, center, width):
        return amplitude * width**2 / ((x - center) ** 2 + width**2)

    start = [values.max(), labels[np.argmax(values)], (labels.max() - labels.min()) / 10]
    return 1e3 * curve_fit(model, labels, values, p0=start)[0][2]


def detector_model(changes, weights=(1.0, 1.0), nu=1.0, delay=5.0):
    """Model of Example 4 with the detector of the Perspective's script, N_exc evolved over the delay.

    N_exc = w_a n_a + w_b n_b - (1 - nu) (a^+ a^+ a a + b^+ b^+ b b); nu = 1 is the plain excitation number.
    """
    operators = models.example4_operators(**changes)
    H, channels = operators["H"], operators["channels"]
    a = qt.tensor(qt.destroy(3), qt.qeye(3))
    b = qt.tensor(qt.qeye(3), qt.destroy(3))
    n_a, n_b = a.dag() * a, b.dag() * b
    number = (weights[0] * n_a + weights[1] * n_b
              - (1 - nu) * (a.dag() * a.dag() * a * a + b.dag() * b.dag() * b * b))
    liouvillian = qt.liouvillian(H, [np.sqrt(rate) * op for op, rate in channels])
    delayed = qt.vector_to_operator((liouvillian.dag() * delay).expm() * qt.operator_to_vector(number))
    detectors = dict(operators["detectors"])
    detectors["N_delayed"] = delayed
    mu_plus = operators["mu_plus"]
    return EigenbasisKModel(H, detectors["polarization"], c_ops_raw=channels, j_plus_array=mu_plus,
                            j_minus_array=mu_plus.dag(), detection_op_array=detectors["polarization"],
                            observable_op_arrays=detectors)


def evaluate(model, labels, observables):
    """Signals of the three pathways at the points (-w, +w), w = physical frequency of the labels."""
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(model)
    w = np.asarray(labels) * LABEL_TO_TRUE
    return solver.generate_spectrum(models.EX4_PROTOCOL, {"omega_1q": -w, "omega_emit": w}, pathways=models.EX4_PATHWAYS,
                                    fixed_coordinates={"t2": T2}, observables=observables)


def pathway_samples(result, observable, pathway):
    return np.diag(result.observables[observable][pathway])


def coherent_rows(changes, labels):
    """R1, R2, R3 (SE, GSB, ESA) and their sum at the sample points, without radiative channels as in the Perspective."""
    result = evaluate(detector_model(dict(changes, k_rad=0.0)), labels, {"polarization": "polarization"})
    rows = [pathway_samples(result, "polarization", p) for p in ("R_SE", "R_GSB", "R_ESA")]
    return rows + [sum(rows)]


def action_rows(changes, labels, **detector):
    """Q3a, Q4a, Q2b, Q8b (FDGF signals, N_exc after the delay), radiative channels included."""
    observables = {k: ObservableSpec.action(k, fourth_interaction=k, operator="N_delayed") for k in ("Bu", "Kd")}
    result = evaluate(detector_model(changes, **detector), labels, observables)
    return [pathway_samples(result, k, p) for _, p, k in DIAGRAMS]


def widths(labels, rows):
    return [lorentzian_halfwidth(labels, np.abs(row)) for row in rows]


def compare_scenario(root, folder, changes):
    data = load_scenario(root, folder)
    report = {"changes": changes, "peaks": {}}
    for key, peak in (("re", "a"), ("re_b", "b")):
        labels = np.asarray(data["X3"][key][2])
        stored_x3 = np.asarray(data["X3"][key][3])                  # R1, R2, R3, sum (|S|, ten points)
        stored_x4 = np.asarray(data["X4"][key][3])                  # Q3a, Q4a, Q2b, Q8b, sum
        coherent = coherent_rows(changes, labels)
        action = action_rows(changes, labels)
        unsigned = sum(row / sign for row, sign in zip(action, RAW_SIGNS))
        physical = sum(action)
        report["peaks"][peak] = {
            "labels_eV": labels.tolist(),
            "X3_stored_meV": [1e3 * g for g in data["X3"][key][4]],
            "X3_fdgf_meV": widths(labels, coherent),
            "X4_stored_meV": [1e3 * g for g in data["X4"][key][4]],
            "X4_fdgf_unsigned_meV": widths(labels, [r / s for r, s in zip(action, RAW_SIGNS)] + [unsigned]),
            "X4_fdgf_physical_meV": widths(labels, action + [physical])[-1],
            "amplitude_ratio_X3": [float(np.mean(np.abs(c) / s)) / AMPLITUDE_SCALE for c, s in zip(coherent, stored_x3)],
            "amplitude_ratio_X4_unsigned": float(np.mean(np.abs(unsigned) / stored_x4[4])) / AMPLITUDE_SCALE,
        }
    return report


def unbalanced_scan(root):
    data = load_scenario(root, "Unbalance")
    out = {"stored_X4_meV": [1e3 * data["X4"][k][4][-1] for k in ("re", "re_b")], "scan": {}}
    for nu in np.arange(0.0, 1.01, 0.1):
        values = []
        for key in ("re", "re_b"):
            labels = np.asarray(data["X3"][key][2])
            rows = action_rows({}, labels, nu=float(nu))
            values.append(widths(labels, [sum(r / s for r, s in zip(rows, RAW_SIGNS))])[0])
        out["scan"][f"{nu:.1f}"] = values
    return out


def projector_widths(root):
    """Complete action signal (Bu + Kd) of the PL projector A_PL of the Perspective's Eq. (30), same estimator."""
    out = {}
    for folder, changes in (("Reference", {}), ("Enhance Transfer trans = 0.01", {"k_transfer": 0.010})):
        data = load_scenario(root, folder)
        entry = {}
        for key, peak in (("re", "a"), ("re_b", "b")):
            labels = np.asarray(data["X3"][key][2])
            solver = SpectroscopySolver(backend="dense", eta=ETA)
            solver.feed_model(models.example4_model(**changes))
            w = labels * LABEL_TO_TRUE
            observables = {name: ObservableSpec.action(name, fourth_interaction=("Bu", "Kd"), operator=name)
                           for name in ("A_PL", "A_PL_unbalanced", "N_exc")}
            result = solver.generate_spectrum(models.EX4_PROTOCOL, {"omega_1q": -w, "omega_emit": w},
                                              pathways=models.EX4_PATHWAYS, fixed_coordinates={"t2": T2},
                                              observables=observables)
            entry[peak] = {name: widths(labels, [sum(pathway_samples(result, name, p) for p in
                                                      ("R_SE", "R_GSB", "R_ESA"))])[0] for name in observables}
            entry[peak]["stored_X4_meV"] = 1e3 * data["X4"][key][4][-1]
        out[folder] = entry
    return out


def main(root):
    results = {
        "grid": {"N": N_POINTS, "dt_true": DT_TRUE, "dt_label": DT_LABEL, "label_step_meV": 1e3 * LABEL_STEP,
                 "amplitude_scale": AMPLITUDE_SCALE},
        "scenarios": {folder: compare_scenario(root, folder, changes) for folder, changes in SCENARIOS.items()},
        "unbalanced": unbalanced_scan(root),
        "projector": projector_widths(root),
    }
    out = models.DATA / "perspective_comparison.json"
    models.results_dirs()
    out.write_text(json.dumps(results, indent=1))
    print(f"{'scenario':<32}{'peak':>5}{'X3 stored':>11}{'X3 FDGF':>9}{'X4 stored':>11}{'X4 FDGF (Perspective signs)':>29}"
          f"{'X4 FDGF (Liouville signs)':>27}")
    for folder, report in results["scenarios"].items():
        for peak, entry in report["peaks"].items():
            print(f"{folder:<32}{peak:>5}{entry['X3_stored_meV'][-1]:>11.3f}{entry['X3_fdgf_meV'][-1]:>9.3f}"
                  f"{entry['X4_stored_meV'][-1]:>11.3f}{entry['X4_fdgf_unsigned_meV'][-1]:>29.3f}"
                  f"{entry['X4_fdgf_physical_meV']:>27.3f}")
    print("written", out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PERSPECTIVE_DATA")
    if not path:
        raise SystemExit("Give the folder 'Data/Borealis Data' of the Perspective repository as argument "
                         "or set PERSPECTIVE_DATA.")
    main(path)
