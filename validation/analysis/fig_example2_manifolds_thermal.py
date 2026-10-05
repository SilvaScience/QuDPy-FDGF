"""Regenerate fig_example2_manifolds_thermal.pdf (Example 2, excitation structure and thermal preparation).

(a) One-magnon dispersion of the six-site XXZ ring at the allowed momenta; the star marks the
    k = 0 state selected by the uniform THz field.
(b) One-to-two-magnon transition strengths from that bright state. The bound-bimagnon transition
    carries most of the accessible strength.
(c) Canonical magnon-sector populations at T = 4 K (read from results/data/example2.npz, saved by
    the example notebook); the dashed line marks the cutoff N_max = 4.

The Hamiltonian blocks and the uniform raising operator come from models.py.

Usage:  python fig_example2_manifolds_thermal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import DATA, EX2, FIGURES, MU_B_MEV_PER_T, example2_chain, results_dirs

results_dirs()
saved = np.load(DATA / "example2.npz")
n_sites, max_magnons = int(saved["n_sites"]), int(saved["max_magnons"])
temperature = float(saved["temperature"])
weights = saved["sector_weight"]

# ------------------------------------------------------------------ blocks of the first sectors
H, N_op, M_minus = example2_chain(n_sites)
magnons = np.rint(N_op.diag().real).astype(int)
members = {N: np.flatnonzero(magnons == N) for N in range(3)}
H_full, M_full = H.full(), M_minus.full()
H1, H2 = (H_full[np.ix_(members[N], members[N])] for N in (1, 2))
M10 = M_full[np.ix_(members[1], members[0])]
M21 = M_full[np.ix_(members[2], members[1])]

one_energies, one_vectors = np.linalg.eigh(H1)
two_energies, two_vectors = np.linalg.eigh(H2)
bright_index = int(np.argmax(np.abs(one_vectors.conj().T @ M10[:, 0]) ** 2))
bright_energy = float(one_energies[bright_index])
amplitudes = two_vectors.conj().T @ M21 @ one_vectors[:, bright_index]
strengths = np.abs(amplitudes) ** 2
energies = two_energies - bright_energy
active = strengths > 1e-10 * strengths.max()
fractions = strengths / strengths.sum()
bound = int(np.flatnonzero(active)[np.argmin(energies[active])])

k_points = 2 * np.pi * np.arange(n_sites) / n_sites
dispersion = EX2["g_factor"] * MU_B_MEV_PER_T * EX2["static_field"] - EX2["J_z"] + EX2["J_xy"] * np.cos(k_points)
print(f"bright k=0 energy {bright_energy:.4f} meV")
print(f"bound-bimagnon transition {energies[bound]:.4f} meV carries {100 * fractions[bound]:.2f} % of the strength")
print(f"population beyond N_max = {weights[max_magnons + 1:].sum():.2e}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)

shown_k = (k_points + np.pi) % (2 * np.pi) - np.pi
order = np.argsort(shown_k)
axes[0].plot(shown_k[order], dispersion[order], "o-", lw=1.5)
axes[0].plot(0.0, bright_energy, "*", ms=12, color="tab:red", label=rf"bright $k=0$: {bright_energy:.3f} meV")
axes[0].set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
axes[0].set_xlabel(r"Momentum $k$")
axes[0].set_ylabel("One-magnon energy (meV)")
axes[0].set_title("One-magnon dispersion")
axes[0].legend(frameon=False, fontsize=8)

markerline, stemlines, _ = axes[1].stem(energies[active], fractions[active], basefmt=" ")
plt.setp(markerline, markersize=5, color="tab:blue")
plt.setp(stemlines, linewidth=1.5, color="tab:blue")
axes[1].plot(energies[bound], fractions[bound], "*", ms=12, color="tab:red", label="bound bimagnon")
axes[1].set_xlabel(r"Emission energy $E_{2,a}-E_{1,\mathrm{bright}}$ (meV)")
axes[1].set_ylabel("Normalized transition strength")
axes[1].set_title("Bright-to-two-magnon transitions")
axes[1].legend(frameon=False, fontsize=8)

sectors = np.arange(weights.size)
axes[2].semilogy(sectors, weights, "o-", lw=1.5)
axes[2].axvline(max_magnons + 0.5, color="0.35", ls="--", lw=1.0, label=rf"cutoff $N_{{\max}}={max_magnons}$")
axes[2].axvspan(max_magnons + 0.5, sectors.max() + 0.5, color="0.88", zorder=-1)
axes[2].set_xticks(sectors)
axes[2].set_xlabel(r"Magnon number $N$")
axes[2].set_ylabel(r"Equilibrium sector weight $P_N$")
axes[2].set_title(rf"Thermal populations at $T={temperature:.1f}$ K")
axes[2].legend(frameon=False, fontsize=8)
for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
    axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

fig.savefig(FIGURES / "fig_example2_manifolds_thermal.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig_example2_manifolds_thermal.png", dpi=200, bbox_inches="tight")
print("saved", FIGURES / "fig_example2_manifolds_thermal.pdf")
