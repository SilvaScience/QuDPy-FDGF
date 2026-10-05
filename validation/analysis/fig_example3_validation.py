"""Regenerate fig_example3_validation.pdf (Example 3, validation of the fifth-order calculation).

(a) Modulus of each pathway at the dominant peak (-E_1, E_1 - eps_1), eta = 8 meV, for the eight
    orderings of the last three interactions (K = Ku, B = Bd; the common pair Bu Bu is omitted),
    at N_max = 2 and N_max = 3. The all-Ku ordering is suppressed by the excitation ceiling at
    N_max = 2; the all-Bd ordering is suppressed by the ground-state floor and never appears.
(b) Emission profile at omega_2Q = -E_1 with eta = 2 meV, normalized to the N_max = 3 maximum.
    Green ticks mark the E^(3)_m - E_k family, which exists only at N_max = 3.
(c) Mode-resolved detection at omega_2Q = -E_1, eta = 8 meV: the total polarization, the two
    mode-resolved dipoles and their sum come from the same propagated pathway states.

The N_max = 3, eta = 8 meV case is read from results/data/example3.npz (saved by the example
notebook); the other cases come from results/data/example3_extra_cases.npz (cases.py).

Usage:  python fig_example3_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py, cases.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cases import example3_extra_cases
from models import DATA, FIGURES, results_dirs

results_dirs()
saved = np.load(DATA / "example3.npz")
extra = example3_extra_cases()
omega_2q, omega_emit = saved["omega_2q"], saved["omega_emit"]
energies = saved["energies"]
epsilon, E_two, E_three = energies[1:3], energies[3:6], energies[6:10]

row = int(np.argmin(np.abs(omega_2q + E_two[0])))
col = int(np.argmin(np.abs(omega_emit - (E_two[0] - epsilon[0]))))
ORDERINGS = ("KKK", "KKB", "KBK", "KBB", "BKK", "BKB", "BBK", "BBB")
PURPLE, GREY = "#7b4ab5", "0.7"


def ordering(label):
    return "".join("K" if step == "Ku" else "B" for step in label.split()[2:])


def peak_amplitudes(labels, maps):
    values = dict.fromkeys(ORDERINGS, 0.0)
    for label, pathway_map in zip(labels, maps):
        values[ordering(str(label))] = abs(pathway_map[row, col])
    return values


amp_n3 = peak_amplitudes(saved["pathway_labels"], saved["pathway_maps"])
amp_n2 = peak_amplitudes(extra["pathway_labels_n2_eta8"], extra["pathway_maps_n2_eta8"])

# ------------------------------------------------------------------ (b) and (c) data
cut_n3 = np.abs(extra["spectrum_n3_eta2"][row])
cut_n2 = np.abs(extra["spectrum_n2_eta2"][row])
total = saved["spectrum"][row]
P_h, P_l = saved["polarization_h"][row], saved["polarization_l"][row]
residual = np.abs(total - (P_h + P_l)).max() / np.abs(total).max()
print(f"all-Ku amplitude: N_max=2 {amp_n2['KKK']:.2e}, N_max=3 {amp_n3['KKK']:.2e}")
print(f"identical orderings: {[k for k in ORDERINGS if amp_n2[k] == amp_n3[k] and amp_n3[k] > 0]}")
print(f"mode-sum residual {residual:.1e}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), constrained_layout=True)

x, width = np.arange(len(ORDERINGS)), 0.4
a = axes[0]
a.bar(x - width / 2, [amp_n2[k] / 1e4 for k in ORDERINGS], width, color=GREY, label=r"$n_{\max}=2$")
a.bar(x + width / 2, [amp_n3[k] / 1e4 for k in ORDERINGS], width, color=PURPLE, label=r"$n_{\max}=3$")
a.set_xticks(x, ORDERINGS, rotation=90)
a.set_ylabel(r"$|S|$ at the peak ($10^4$)")
a.set_title("(a) orderings")
a.set_ylim(0, 6.8)
a.text(-width / 2, 0.15, "ceiling", rotation=90, ha="center", va="bottom", color="0.4", fontsize=7)
a.text(len(ORDERINGS) - 1, 0.15, "floor", rotation=90, ha="center", va="bottom", color="0.4", fontsize=7)
a.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2)

b = axes[1]
b.plot(omega_emit, cut_n2 / cut_n3.max(), lw=3.0, color=GREY, label=r"$n_{\max}=2$")
b.plot(omega_emit, cut_n3 / cut_n3.max(), lw=1.5, color=PURPLE, label=r"$n_{\max}=3$")
ticks = (E_three[:, None] - E_two[None, :]).ravel()
b.vlines(ticks[(ticks >= omega_emit[0]) & (ticks <= omega_emit[-1])], 0, 0.1,
         transform=b.get_xaxis_transform(), color="#2ca02c", lw=1.2)
b.set_xlim(omega_emit[0], omega_emit[-1])
b.set_xlabel(r"$\omega_{\rm emit}$ (eV)")
b.set_ylabel(r"$|S|$ at $\omega_{2Q}=-E_1$")
b.set_title(r"(b) truncation, $\eta=2$ meV")
b.legend(frameon=False, fontsize=7, loc="upper left")

c = axes[2]
c.plot(omega_emit, np.abs(total) / 1e5, lw=3.0, color=GREY, label=r"$P$")
c.plot(omega_emit, np.abs(P_h + P_l) / 1e5, "k--", lw=1.3, label=r"$P_h+P_l$")
c.plot(omega_emit, np.abs(P_h) / 1e5, color="#d62728", lw=0.9, label=r"$P_h$")
c.plot(omega_emit, np.abs(P_l) / 1e5, color="#1f77b4", lw=0.9, label=r"$P_l$")
c.set_xlim(omega_emit[0], omega_emit[-1])
c.set_xlabel(r"$\omega_{\rm emit}$ (eV)")
c.set_ylabel(r"$|P|$ ($10^5$), $\omega_{2Q}=-E_1$")
c.set_title(r"(c) modes, $\eta=8$ meV")
c.legend(frameon=False, fontsize=7, loc="upper right")

fig.savefig(FIGURES / "fig_example3_validation.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig_example3_validation.png", dpi=200, bbox_inches="tight")
print("saved", FIGURES / "fig_example3_validation.pdf")
