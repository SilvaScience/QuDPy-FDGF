"""Regenerate fig_example1_manifolds_thermal.pdf (Example 1, excitation structure and thermal preparation).

(a) One-magnon dispersion of the six-site XXZ ring: the analytic curve and the allowed momenta; the star
    marks the k = 0 state selected by the uniform THz field.
(b) Two-magnon spectrum as a function of the total momentum K: the levels of the six-site ring (the lowest
    level of each K is the bound branch), the lower and upper edges of the two-magnon continuum of the infinite
    chain, and the bound state of the infinite chain, E(K) = 2 eps + V + 4 t^2 cos^2(K/2) / V with
    eps = g mu_B B_0 - J_z, t = J_xy / 2, V = J_z (valid for |J_z| > |J_xy|). The arrow marks the binding
    energy at K = 0 for the infinite chain, 2 omega_1(0) - E(0).
(c) One-to-two-magnon transition strengths from the bright state. The bound-bimagnon transition carries most
    of the accessible strength.
(d) Canonical magnon-sector populations at T = 4 K (read from results/data/example1.npz, saved by the
    example notebook); the dashed line marks the cutoff N_max = 4.

The Hamiltonian blocks and the uniform raising operator come from models.py. The levels of (b) are obtained by
projecting the two-magnon sector on each total momentum K = 2 pi m / L with the translation operator.

Usage:  python fig_example1_manifolds_thermal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import DATA, EX1, FIGURES, MU_B_MEV_PER_T, example1_chain, results_dirs

results_dirs()
saved = np.load(DATA / "example1.npz")
n_sites, max_magnons = int(saved["n_sites"]), int(saved["max_magnons"])
temperature = float(saved["temperature"])
weights = saved["sector_weight"]

# ------------------------------------------------------------------ blocks of the first sectors
H, N_op, M_minus = example1_chain(n_sites)
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

gmuB = EX1["g_factor"] * MU_B_MEV_PER_T * EX1["static_field"]
epsilon, hop, V = gmuB - EX1["J_z"], EX1["J_xy"] / 2.0, EX1["J_z"]            # one-magnon offset, hopping amplitude, nn interaction
k_points = 2 * np.pi * np.arange(n_sites) / n_sites
dispersion = gmuB - EX1["J_z"] + EX1["J_xy"] * np.cos(k_points)

# ------------------------------------------------------------------ two-magnon levels at each total momentum K = 2 pi m / L
shift = np.zeros((2 ** n_sites, 2 ** n_sites))                                  # translation of the ring on the spin basis
for state in range(2 ** n_sites):
    bits = [(state >> (n_sites - 1 - site)) & 1 for site in range(n_sites)]
    moved = bits[-1:] + bits[:-1]
    shift[sum(bit << (n_sites - 1 - site) for site, bit in enumerate(moved)), state] = 1.0
T2 = shift[np.ix_(members[2], members[2])]
levels_by_K = {}
for m in range(n_sites):
    K = 2 * np.pi * m / n_sites
    projector = sum(np.exp(-1j * K * n) * np.linalg.matrix_power(T2, n) for n in range(n_sites)) / n_sites
    values, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
    basis = vectors[:, values > 0.5]
    levels_by_K[(K + np.pi) % (2 * np.pi) - np.pi] = np.linalg.eigvalsh(basis.conj().T @ H2 @ basis)

lowest_K0 = float(levels_by_K[0.0].min())
threshold_K0 = 2 * (epsilon + 2 * hop)                                          # 2 omega_1(0) = lower edge of the K = 0 continuum
bound_K0_infinite = 2 * epsilon + V + 4 * hop ** 2 / V
assert abs(lowest_K0 - float(two_energies.min())) < 1e-9 or lowest_K0 > two_energies.min() - 1e-9
print(f"bright k=0 energy {bright_energy:.4f} meV")
print(f"bound-bimagnon transition {energies[bound]:.4f} meV carries {100 * fractions[bound]:.2f} % of the strength")
print(f"K=0 levels of the ring: {np.round(np.sort(levels_by_K[0.0]), 4)}; infinite chain: bound {bound_K0_infinite:.4f}, "
      f"threshold {threshold_K0:.4f}, binding {threshold_K0 - bound_K0_infinite:.4f} meV")
print(f"population beyond N_max = {weights[max_magnons + 1:].sum():.2e}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True)
ax_disp, ax_two, ax_strength, ax_pop = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# (a) one-magnon dispersion
k_line = np.linspace(-np.pi, np.pi, 400)
ax_disp.plot(k_line, gmuB - EX1["J_z"] + EX1["J_xy"] * np.cos(k_line), "-", lw=1.5, color="tab:blue", zorder=1)
shown_k = (k_points + np.pi) % (2 * np.pi) - np.pi
ax_disp.plot(shown_k, dispersion, "o", ms=6, color="tab:blue", zorder=2, label=f"allowed momenta, $L={n_sites}$")
ax_disp.plot(0.0, bright_energy, "*", ms=13, color="tab:red", zorder=3, label=rf"bright $k=0$: {bright_energy:.3f} meV")
ax_disp.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
ax_disp.set_xlabel(r"Momentum $k$")
ax_disp.set_ylabel("One-magnon energy (meV)")
ax_disp.set_title("One-magnon dispersion")
ax_disp.legend(frameon=False, fontsize=8, loc="upper center")

# (b) two-magnon spectrum versus total momentum
K_line = np.linspace(-np.pi, np.pi, 400)
half_width = 4 * abs(hop) * np.abs(np.cos(K_line / 2))
ax_two.fill_between(K_line, 2 * epsilon - half_width, 2 * epsilon + half_width, color="0.88", zorder=0,
                    label="continuum, $L=\\infty$")
ax_two.plot(K_line, 2 * epsilon + V + 4 * hop ** 2 * np.cos(K_line / 2) ** 2 / V, "-", color="tab:red", lw=1.4,
            zorder=1, label="bound state, $L=\\infty$")
first = True
for K, levels in sorted(levels_by_K.items()):
    levels = np.sort(levels)
    ax_two.plot([K] * (len(levels) - 1), levels[1:], "o", mfc="none", mec="0.35", ms=5, zorder=2,
                label=f"levels, $L={n_sites}$" if first else None)
    ax_two.plot(K, levels[0], "o", color="tab:red", ms=5, zorder=3, label=f"lowest level, $L={n_sites}$" if first else None)
    first = False
ax_two.plot(0.0, lowest_K0, "*", ms=11, color="tab:red", zorder=4)
ax_two.annotate("", xy=(0.0, bound_K0_infinite), xytext=(0.0, threshold_K0),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.9, shrinkA=0, shrinkB=0), zorder=5)
ax_two.annotate(rf"binding {1e3 * (threshold_K0 - bound_K0_infinite):.0f} $\mu$eV", xy=(0.0, 0.5 * (bound_K0_infinite + threshold_K0)),
                xytext=(0.55, 2.52), fontsize=8, arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
ax_two.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
ax_two.set_xlabel(r"Total momentum $K$")
ax_two.set_ylabel("Two-magnon energy (meV)")
ax_two.set_title("Two-magnon spectrum")
ax_two.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)

# (c) transition strengths from the bright state
markerline, stemlines, _ = ax_strength.stem(energies[active], fractions[active], basefmt=" ")
plt.setp(markerline, markersize=5, color="tab:blue")
plt.setp(stemlines, linewidth=1.5, color="tab:blue")
ax_strength.plot(energies[bound], fractions[bound], "*", ms=13, color="tab:red", label="bound bimagnon")
ax_strength.set_xlabel(r"Emission energy $E_{2,a}-E_{1,\mathrm{bright}}$ (meV)")
ax_strength.set_ylabel("Normalized transition strength")
ax_strength.set_title("Bright-to-two-magnon transitions")
ax_strength.legend(frameon=False, fontsize=8)

# (d) thermal populations
sectors = np.arange(weights.size)
ax_pop.semilogy(sectors, weights, "o-", lw=1.5)
ax_pop.axvline(max_magnons + 0.5, color="0.35", ls="--", lw=1.0, label=rf"cutoff $N_{{\max}}={max_magnons}$")
ax_pop.axvspan(max_magnons + 0.5, sectors.max() + 0.5, color="0.88", zorder=-1)
ax_pop.set_xticks(sectors)
ax_pop.set_xlabel(r"Magnon number $N$")
ax_pop.set_ylabel(r"Equilibrium sector weight $P_N$")
ax_pop.set_title(rf"Thermal populations at $T={temperature:.1f}$ K")
ax_pop.legend(frameon=False, fontsize=8)

for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
    axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

fig.savefig(FIGURES / "fig_example1_manifolds_thermal.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig_example1_manifolds_thermal.png", dpi=200, bbox_inches="tight")
print("saved", FIGURES / "fig_example1_manifolds_thermal.pdf")
