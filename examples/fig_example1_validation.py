"""Regenerate fig_example1_validation.pdf (Example 1, open two-level system).

(a) peak amplitude of R1 (SE) and R2 (GSB) versus t2, normalized to t2 = 0;
(b) N_rad^(4)(T) / P_e^(4) for R1 versus the detection window T;
(c) relative quadrature error of (b) at T = 5/gamma_1 versus n_steps.

The model is the one of examples/two_level_open_observables.ipynb.
Labels use gamma_1 = 1/T1 (population decay rate) as in the revised manuscript.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qudpy_fdgf import EigenbasisKModel, ObservableSpec, SpectroscopySolver
from qudpy_fdgf.pathways import FrequencyPathway
from qudpy_fdgf.protocols import standard_nq_protocol

OUT = Path(__file__).resolve().parent / "Figures"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ model (Example 1)
omega_eg, dipole = 2.0, 0.5
gamma_1, gamma_phi = 0.04, 0.03
H = np.diag([0.0, omega_eg]).astype(complex)
mu = dipole * np.array([[0, 1], [1, 0]], dtype=complex)
P_g = np.diag([1.0, 0.0]).astype(complex)
P_e = np.diag([0.0, 1.0]).astype(complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_z = np.diag([-1.0, 1.0]).astype(complex)

model = EigenbasisKModel(
    H, mu,
    c_ops_raw=((sigma_minus, gamma_1), (sigma_z, gamma_phi / 2.0)),
    observable_op_arrays={"polarization": mu, "ground_population": P_g,
                          "excited_population": P_e},
)
solver = SpectroscopySolver(backend="dense", eta=0.002)
solver.feed_model(model)
radiative = solver.jump_channel_names()[0]

R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"),
                      component="rephasing", detection="polarization")
R2 = FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"),
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
amp = {"R1": [], "R2": []}
for t2 in t2_grid:
    res = at_peak(t2, (R1, R2))
    for name in amp:
        amp[name].append(res.pathways[name][0, 0])
amp = {k: np.abs(np.array(v)) / abs(v[0]) for k, v in amp.items()}
r1_at_10 = abs(at_peak(10.0, (R1,)).pathways["R1"][0, 0]) / abs(at_peak(0.0, (R1,)).pathways["R1"][0, 0])
err_a = np.max(np.abs(amp["R1"] - np.exp(-gamma_1 * t2_grid)))
print(f"(a) R1(t2=10)/R1(0) = {r1_at_10:.6f}   (exp(-0.4) = {np.exp(-0.4):.6f})")
print(f"(a) max |R1 - exp(-gamma_1 t2)| = {err_a:.1e};  R2 spread = {np.ptp(amp['R2']):.1e}")


# ------------------------------------------------------------------ (b), (c) integrated fluorescence
def fluorescence_ratio(T, n_steps=101):
    obs = {
        "pop": ObservableSpec.action("pop", fourth_interaction="Bu", operator="excited_population"),
        "fl": ObservableSpec.mean_jump("fl", radiative, time_window=(0.0, T),
                                       n_steps=n_steps, fourth_interaction="Bu"),
    }
    res = at_peak(10.0, (R1,), obs)
    return (res.observables["fl"]["R1"][0, 0] / res.observables["pop"]["R1"][0, 0]).real


T_grid = np.linspace(0.0, 200.0, 41)[1:]
ratio = np.array([fluorescence_ratio(T) for T in T_grid])
T_used = 5.0 / gamma_1
exact = 1.0 - np.exp(-gamma_1 * T_used)

n_grid = np.unique(np.round(np.logspace(1, np.log10(2000), 14)).astype(int))
n_grid = np.union1d(n_grid, [101])
rel_err = np.array([abs(fluorescence_ratio(T_used, n) - exact) / exact for n in n_grid])
err_101 = rel_err[n_grid == 101][0]
slope = np.polyfit(np.log(n_grid[n_grid >= 30]), np.log(rel_err[n_grid >= 30]), 1)[0]
print(f"(b) T = 5/gamma_1 = {T_used:.0f} eV^-1;  max |ratio - (1 - exp(-gamma_1 T))| = "
      f"{np.max(np.abs(ratio - (1 - np.exp(-gamma_1 * T_grid)))):.1e}")
print(f"(c) relative error at n_steps = 101: {err_101:.2e};  fitted slope = {slope:.2f}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.titlesize": 7,
                     "lines.linewidth": 1.0})
fig, ax = plt.subplots(1, 3, figsize=(390 / 72, 140.4 / 72), constrained_layout=True)

a = ax[0]
a.plot(t2_grid, amp["R2"], color="C0", lw=1.6, label=r"$R_2$ (GSB)")
a.plot(t2_grid, np.ones_like(t2_grid), color="0.3", lw=0.8, ls="--", label="stationary")
a.plot(t2_grid[::3], amp["R1"][::3], "o", ms=2.2, color="C3", label=r"$R_1$ (SE)")
a.plot(t2_grid, np.exp(-gamma_1 * t2_grid), color="C3", lw=0.8, label=r"$e^{-\gamma_1 t_2}$")
a.axvline(10.0, color="k", ls=":", lw=0.8)
a.set(xlim=(0, 60), ylim=(0, 1.1), xlabel=r"$t_2$ (eV$^{-1}$)", ylabel="peak amplitude",
      title="(a) waiting time")
a.legend(loc="center right", frameon=False, handlelength=1.5)

b = ax[1]
T_fine = np.linspace(0, 200, 401)
b.plot(T_fine, 1 - np.exp(-gamma_1 * T_fine), color="0.3", lw=0.8, label=r"$1-e^{-\gamma_1 T}$")
b.plot(T_grid, ratio, "o", ms=2.2, color="C2", label="numerical")
b.axvline(T_used, color="k", ls=":", lw=0.8, label=r"$T=5/\gamma_1$")
b.set(xlim=(0, 200), ylim=(0, 1.1), xlabel=r"$T$ (eV$^{-1}$)",
      ylabel=r"$N_{\mathrm{rad}}^{(4)}/P_e^{(4)}$", title="(b) fluorescence window")
b.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="none", handlelength=1.5)

c = ax[2]
c.loglog(n_grid, rel_err, "o", ms=2.2, color="C4", label="observed")
c.loglog(n_grid, err_101 * (n_grid / 101.0) ** -2, color="0.3", lw=0.8, ls="--", label=r"$\propto n^{-2}$")
c.set(xlabel=r"$n_{\mathrm{steps}}$", ylabel="relative error", title="(c) quadrature")
c.legend(loc="upper right", frameon=False, handlelength=1.5)

fig.savefig(OUT / "fig_example1_validation.pdf")
fig.savefig(OUT / "fig_example1_validation.png", dpi=200)
print("saved", OUT / "fig_example1_validation.pdf")
