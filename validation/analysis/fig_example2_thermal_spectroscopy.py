"""Regenerate fig_example2_thermal_spectroscopy.pdf (Example 2, thermal XXZ chain).

Complete third-order rephasing response of the six-site XXZ chain at 4 K: the eight
instruction sequences with coherence orders q = (-1, 0, +1). Three of them (GSB, SE, ESA)
are active for a pure ground state; the five others start with a lowering interaction or
end with two bra lowerings and contribute only from thermally populated sectors.

(a) thermal part of the response, S(4 K) - P_0 S(0 K): the sector N = 0 holds only the
    polarized ground state, so this is exactly the contribution of the initial sectors N >= 1;
(b) ESA pathway at 4 K;
(c) complete rephasing response at 4 K;
(d) emission cut at the bright excitation pole: GSB + SE, ESA, the complete response at 4 K,
    and the complete response of the pure ground state.

Model and grids are those of the manuscript (N_max = 4, D = 57, eta = 0.03 meV, 60 x 60
grid, t2 = 0). The sparse backend uses the diagonal preconditioner, which is exact for this
closed model in its eigenbasis (one GMRES iteration per resolvent). Spectra are stored in
results/data/fig_example2_thermal_spectroscopy.npz; ``python fig_example2_thermal_spectroscopy.py
plot`` redraws the figure from that file.
"""
import sys
import time
from itertools import product
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py)
from models import FIGURES, DATA as DATA_DIR, results_dirs

results_dirs()
from models import BOLTZMANN_MEV_PER_K, EX2, example2_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qudpy_fdgf import (FrequencyPathway, SpectroscopySolver,
                        ThermodynamicContext, standard_nq_protocol)

OUT = FIGURES
DATA = DATA_DIR / "fig_example2_thermal_spectroscopy.npz"

# ------------------------------------------------------------------ model (Example 2, models.py)
n_sites, max_magnons = EX2["n_sites"], EX2["max_magnons"]
temperature, eta = EX2["temperature"], EX2["eta"]                  # K, meV
K_B = BOLTZMANN_MEV_PER_K
model, hamiltonians = example2_model()

energies = {N: np.linalg.eigvalsh(H) for N, H in hamiltonians.items()}
bright = energies[1].min()                                   # k = 0 magnon, 1.1224 meV
bimagnon = energies[2].min() - bright                        # 0.9884 meV
weights = {N: np.exp(-e / (K_B * temperature)).sum() for N, e in energies.items()}
P_0 = weights[0] / sum(weights.values())                     # ground-state population at 4 K

# ------------------------------------------------------------------ pathways and protocol
ZTC = {("Bu", "Bd", "Ku"): "GSB", ("Bu", "Ku", "Bd"): "SE", ("Bu", "Ku", "Ku"): "ESA"}
SEQUENCES = tuple(product(("Bu", "Kd"), ("Ku", "Bd"), ("Ku", "Bd")))   # q = (-1, 0, +1)
PATHWAYS = tuple(FrequencyPathway(name=ZTC.get(seq, "".join(seq)), interactions=seq,
                                  component="rephasing") for seq in SEQUENCES)
THERMAL_ONLY = tuple(p.name for p in PATHWAYS if p.name not in ZTC.values())
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                nq_axis="omega_1q", detection_axis="omega_emit")
AXES = {"omega_1q": np.linspace(-1.55, -0.55, 60), "omega_emit": np.linspace(0.55, 1.55, 60)}


def spectrum(context, axes=AXES):
    solver = SpectroscopySolver(backend="sparse_sector", eta=eta, krylov_tolerance=1e-11,
                                preconditioner="diagonal")
    solver.feed_model(model, context=context)
    start = time.perf_counter()
    result = solver.generate_spectrum(PROTOCOL, axes, pathways=PATHWAYS,
                                      fixed_coordinates={"t2": 0.0})
    print(f"  {len(PATHWAYS)} pathways on {axes['omega_1q'].size} x "
          f"{axes['omega_emit'].size}: {time.perf_counter() - start:.0f} s", flush=True)
    return {name: np.asarray(value) for name, value in result.pathways.items()}


def compute():
    print(f"D = {sum(H.shape[0] for H in hamiltonians.values())}, P_0 = {P_0:.6f}", flush=True)
    print("4 K:", flush=True)
    thermal = spectrum(ThermodynamicContext(temperature=temperature))
    print("0 K (pure ground state):", flush=True)
    ground = spectrum(None)
    np.savez(DATA, **{f"thermal_{k}": v for k, v in thermal.items()},
             **{f"ground_{k}": v for k, v in ground.items()}, P_0=P_0)
    return thermal, ground


def load():
    data = np.load(DATA)
    thermal = {k[8:]: data[k] for k in data.files if k.startswith("thermal_")}
    ground = {k[7:]: data[k] for k in data.files if k.startswith("ground_")}
    return thermal, ground


thermal, ground = load() if (len(sys.argv) > 1 and sys.argv[1] == "plot") else compute()
w1, w3 = AXES["omega_1q"], AXES["omega_emit"]
total_T = sum(thermal.values())
total_0 = sum(ground.values())
hot = total_T - P_0 * total_0
ztc_hot = sum(thermal[n] for n in ZTC.values()) - P_0 * sum(ground[n] for n in ZTC.values())
complement = sum(thermal[n] for n in THERMAL_ONLY)

# ------------------------------------------------------------------ checks
scale = np.abs(total_T).max()
residual_0 = max(np.abs(ground[n]).max() for n in THERMAL_ONLY) / np.abs(total_0).max()
i, j = np.unravel_index(np.abs(total_T).argmax(), total_T.shape)
h_i, h_j = np.unravel_index(np.abs(hot).argmax(), hot.shape)
print(f"thermal-only pathways at 0 K: max / max|S(0 K)| = {residual_0:.1e}")
print(f"strongest feature at 4 K: ({w1[i]:.4f}, {w3[j]:.4f}) meV")
print(f"thermal part: max = {np.abs(hot).max() / scale:.3f} max|S(4 K)| at "
      f"({w1[h_i]:.4f}, {w3[h_j]:.4f}) meV; expected (-{bimagnon:.4f}, {bimagnon:.4f})")
print(f"  from GSB/SE/ESA: {np.abs(ztc_hot).max() / scale:.3f}, from the five thermal-only "
      f"pathways: {np.abs(complement).max() / scale:.3f} (fractions of max|S(4 K)|)")
assert residual_0 < 1e-12
assert abs(w1[h_i] + bimagnon) < 0.02 and abs(w3[h_j] - bimagnon) < 0.03

# ------------------------------------------------------------------ figure
extent = [w3[0], w3[-1], w1[0], w1[-1]]
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)


def label(ax, text):
    ax.text(-0.14, 1.06, text, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


def map_panel(ax, values, title, colorbar_label, guides=False):
    image = ax.imshow(np.abs(values), origin="lower", extent=extent, aspect="auto",
                      cmap="magma", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(r"Emission energy $\omega_{\mathrm{emit}}$ (meV)")
    ax.set_ylabel(r"Excitation energy $\omega_{1Q}$ (meV)")
    if guides:
        for x in (bimagnon, bright):
            ax.axvline(x, color="w", ls=":", lw=0.8)
        for y in (-bimagnon, -bright):
            ax.axhline(y, color="w", ls=":", lw=0.8)


map_panel(axes[0, 0], hot, r"Thermal contribution, $N\geq1$", r"$|S-P_0S_{0\,\mathrm{K}}|$",
          guides=True)
map_panel(axes[0, 1], thermal["ESA"], "Bimagnon ESA", r"$|R_{\mathrm{ESA}}|$")
map_panel(axes[1, 0], total_T, "Total rephasing response", r"$|S_{\mathrm{R}}|$")

cut = np.argmin(np.abs(w1 + bright))
ax = axes[1, 1]
ax.plot(w3, np.abs(thermal["GSB"][cut] + thermal["SE"][cut]), label="GSB + SE")
ax.plot(w3, np.abs(thermal["ESA"][cut]), label="ESA")
ax.plot(w3, np.abs(total_T[cut]), "k", lw=2, label=f"total, {temperature:.0f} K")
ax.plot(w3, np.abs(total_0[cut]), "k--", lw=1, label="total, 0 K")
ax.axvline(bright, color="0.4", ls="--", lw=0.8)
ax.axvline(bimagnon, color="C3", ls=":", lw=0.8)
ax.set_title(rf"Cut at $\omega_{{1Q}}={w1[cut]:.3f}$ meV")
ax.set_xlabel(r"Emission energy $\omega_{\mathrm{emit}}$ (meV)")
ax.set_ylabel("Absolute response (solver units)")
ax.legend(fontsize=8)
for ax, text in zip(axes.flat, "abcd"):
    label(ax, f"({text})")

fig.savefig(OUT / "fig_example2_thermal_spectroscopy.pdf")
fig.savefig(OUT / "fig_example2_thermal_spectroscopy.png", dpi=200)
print("saved", OUT / "fig_example2_thermal_spectroscopy.pdf")
