"""Regenerate fig_example1_validation.pdf (Example 1, open two-level system).

(a) peak amplitude of R1 (SE), R2 (GSB) and R3 versus t2, in units of the t2 = 0 amplitude of R2,
    and the sum (R1 + R2 + R3) / 2; R3 = (Bu, Ku, Ku) is non-zero only because the radiative channel
    refills the ground state during t2;
(b) N_rad^(4)(T) / P_e^(4) for R1 versus the detection window T;
(c) relative error of (b) at T = 5/gamma_1: legacy trapezoidal option versus n_steps,
    and the default exact (Heisenberg-picture) evaluation.

The model is the one of examples/example1_two_level_open.ipynb (validation/models.py).
Labels use gamma_1 = 1/T1 (population decay rate) as in the revised manuscript.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py)
from models import FIGURES, DATA as DATA_DIR, results_dirs

results_dirs()
from models import EX1, example1_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qudpy_fdgf import ObservableSpec, SpectroscopySolver
from qudpy_fdgf.pathways import FrequencyPathway
from qudpy_fdgf.protocols import standard_nq_protocol

OUT = FIGURES

# ------------------------------------------------------------------ model (Example 1, models.py)
model = example1_model()
omega_eg, gamma_1, gamma_phi = EX1["omega_eg"], EX1["gamma_1"], EX1["gamma_phi"]
solver = SpectroscopySolver(backend="dense", eta=EX1["eta"])
solver.feed_model(model)
radiative = solver.jump_channel_names()[0]

R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"),
                      component="rephasing", detection="polarization")
R2 = FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"),
                      component="rephasing", detection="polarization")
R3 = FrequencyPathway(name="R3", interactions=("Bu", "Ku", "Ku"),
                      component="rephasing", detection="polarization")
protocol = standard_nq_protocol(order=1, nq_interval=1, detection_interval=3,
                                n_interactions=3, nq_axis="omega_1q",
                                detection_axis="omega_emit")
peak = {"omega_1q": np.array([-2.0]), "omega_emit": np.array([2.0])}


def at_peak(t2, pathways, observables=None):
    return solver.generate_spectrum(protocol, peak, fixed_coordinates={"t2": t2},
                                    pathways=pathways, observables=observables)


# ------------------------------------------------------------------ (a) waiting time
t2_grid = np.linspace(0.0, 60.0, 121)
peak_values = {"R1": [], "R2": [], "R3": []}
for t2 in t2_grid:
    res = at_peak(t2, (R1, R2, R3))
    for name in peak_values:
        peak_values[name].append(res.pathways[name][0, 0])
unit = peak_values["R2"][0]                       # R2 does not depend on t2
rel = {k: np.array(v) / unit for k, v in peak_values.items()}
assert max(np.abs(v.imag).max() for v in rel.values()) < 1e-12        # same phase for the three pathways
rel = {k: v.real for k, v in rel.items()}
total_half = (rel["R1"] + rel["R2"] + rel["R3"]) / 2
decay = np.exp(-gamma_1 * t2_grid)
err_a = {"R1": np.max(np.abs(rel["R1"] - decay)), "R2": np.max(np.abs(rel["R2"] - 1.0)),
         "R3": np.max(np.abs(rel["R3"] + 1.0 - decay)), "total": np.max(np.abs(total_half - decay))}
print(f"(a) R1(t2=10)/R2(0) = {rel['R1'][t2_grid == 10.0][0]:.6f}   (exp(-0.4) = {np.exp(-0.4):.6f})")
print("(a) max deviations: " + ", ".join(f"{k} {v:.1e}" for k, v in err_a.items()))
print(f"(a) (R1+R2)/(R1+R2+R3) at t2 = 10: {(rel['R1'] + rel['R2'])[t2_grid == 10.0][0] / (2 * decay[t2_grid == 10.0][0]):.4f}")


# ------------------------------------------------------------------ (b), (c) integrated fluorescence
def fluorescence_ratio(T, n_steps=101, integration="exact"):
    obs = {
        "pop": ObservableSpec.action("pop", fourth_interaction="Bu", operator="excited_population"),
        "fl": ObservableSpec.mean_jump("fl", radiative, time_window=(0.0, T), n_steps=n_steps,
                                       fourth_interaction="Bu", integration=integration),
    }
    res = at_peak(10.0, (R1,), obs)
    return (res.observables["fl"]["R1"][0, 0] / res.observables["pop"]["R1"][0, 0]).real


T_grid = np.linspace(0.0, 200.0, 41)[1:]
ratio = np.array([fluorescence_ratio(T) for T in T_grid])
T_used = 5.0 / gamma_1
exact = 1.0 - np.exp(-gamma_1 * T_used)

n_grid = np.unique(np.round(np.logspace(1, np.log10(2000), 14)).astype(int))
n_grid = np.union1d(n_grid, [101])
rel_err = np.array([abs(fluorescence_ratio(T_used, n, "trapezoid") - exact) / exact for n in n_grid])
err_101 = rel_err[n_grid == 101][0]
slope = np.polyfit(np.log(n_grid[n_grid >= 30]), np.log(rel_err[n_grid >= 30]), 1)[0]
err_exact = abs(fluorescence_ratio(T_used) - exact) / exact
print(f"(b) T = 5/gamma_1 = {T_used:.0f} eV^-1;  max |ratio - (1 - exp(-gamma_1 T))| = "
      f"{np.max(np.abs(ratio - (1 - np.exp(-gamma_1 * T_grid)))):.1e}  (exact evaluation)")
print(f"(c) trapezoid: relative error at n_steps = 101: {err_101:.2e};  fitted slope = {slope:.2f};"
      f"  exact evaluation: {err_exact:.1e}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.titlesize": 7,
                     "lines.linewidth": 1.0})
fig, ax = plt.subplots(1, 3, figsize=(390 / 72, 140.4 / 72), constrained_layout=True)

a = ax[0]
a.plot(t2_grid, np.ones_like(t2_grid), color="C0", lw=1.4)
a.plot(t2_grid[::3], rel["R1"][::3], "o", ms=2.2, color="C3")
a.plot(t2_grid, decay, color="C3", lw=0.8)
a.plot(t2_grid[::3], rel["R3"][::3], "^", ms=2.4, color="C1")
a.plot(t2_grid, -(1.0 - decay), color="C1", lw=0.8)
a.plot(t2_grid[1::3], total_half[1::3], "x", ms=2.6, color="k", mew=0.7)
a.axhline(0.0, color="0.7", lw=0.5)
a.axvline(10.0, color="k", ls=":", lw=0.8)
a.text(59, 1.045, r"$R_2$ (GSB)", color="C0", ha="right", va="bottom", fontsize=6)
a.text(59, 0.62, r"$R_1$ (SE)," + chr(10) + r"$(R_1+R_2+R_3)/2$:" + chr(10) + r"$e^{-\gamma_1 t_2}$", color="0.2",
       ha="right", va="center", fontsize=6, linespacing=1.3)
a.text(59, -0.50, r"$R_3$: $-(1-e^{-\gamma_1 t_2})$", color="C1", ha="right", va="center", fontsize=6)
a.set(xlim=(0, 60), ylim=(-1.1, 1.25), xlabel=r"$t_2$ (eV$^{-1}$)", ylabel=r"peak amplitude / $R_2(0)$",
      title="(a) waiting time")

b = ax[1]
T_fine = np.linspace(0, 200, 401)
b.plot(T_fine, 1 - np.exp(-gamma_1 * T_fine), color="0.3", lw=0.8, label=r"$1-e^{-\gamma_1 T}$")
b.plot(T_grid, ratio, "o", ms=2.2, color="C2", label="numerical")
b.axvline(T_used, color="k", ls=":", lw=0.8, label=r"$T=5/\gamma_1$")
b.set(xlim=(0, 200), ylim=(0, 1.1), xlabel=r"$T$ (eV$^{-1}$)",
      ylabel=r"$N_{\mathrm{rad}}^{(4)}/P_e^{(4)}$", title="(b) fluorescence window")
b.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="none", handlelength=1.5)

c = ax[2]
floor = 1e-16                                    # display level of a machine-precision error
c.loglog(n_grid, rel_err, "o", ms=2.2, color="C4", label="trapezoid")
c.loglog(n_grid, err_101 * (n_grid / 101.0) ** -2, color="0.3", lw=0.8, ls="--", label=r"$\propto n^{-2}$")
c.axhline(max(err_exact, floor), color="C2", lw=1.2, label="exact (default)")
c.set(xlabel=r"$n_{\mathrm{steps}}$", ylabel="relative error", title="(c) window integral",
      ylim=(floor / 10, 1.0))
c.legend(loc="center right", frameon=False, handlelength=1.5)

fig.savefig(OUT / "fig_example1_validation.pdf")
fig.savefig(OUT / "fig_example1_validation.png", dpi=200)
print("saved", OUT / "fig_example1_validation.pdf")
