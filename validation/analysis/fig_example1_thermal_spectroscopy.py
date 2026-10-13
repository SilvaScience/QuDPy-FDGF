"""Regenerate fig_example1_thermal_spectroscopy.pdf (Example 1, thermal XXZ chain).

Complete third-order rephasing response of the six-site XXZ chain at 4 K: the eight
instruction sequences with coherence orders q = (-1, 0, +1). Three of them (GSB, SE, ESA)
are active for a pure ground state; the five others start with a lowering interaction or
end with two bra lowerings and contribute only from thermally populated sectors.

Top row, modulus: (a) thermal part of the response, S(4 K) - P_0 S(0 K): the sector N = 0 holds
only the polarized ground state, so this is exactly the contribution of the initial sectors
N >= 1; (b) ESA pathway at 4 K; (c) complete rephasing response at 4 K.
Second row, (d)-(f): real part of i S (the display convention of the article, GSB and SE
positive, ESA negative) of the same three quantities. The dashed line of the maps is the
antidiagonal omega_emit = -omega_1Q.
(g) emission cut at the bright excitation pole: GSB + SE (positive), ESA (negative), the complete
    response at 4 K, and the complete response of the pure ground state. Amplitudes are in units
    of 1 / eta^2.

Model and grids are those of the manuscript (N_max = 4, D = 57, eta = 0.03 meV, 60 x 60
grid, t2 = 0). The sparse backend uses the diagonal preconditioner, which is exact for this
closed model in its eigenbasis (one GMRES iteration per resolvent). The checks and the numbers
of the article come from the 60 x 60 grid, stored in
results/data/fig_example1_thermal_spectroscopy.npz (``python fig_example1_thermal_spectroscopy.py
plot`` reads that file instead of recomputing it). The maps themselves are drawn on a finer
181 x 181 grid, computed at each run and not stored.
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
from models import BOLTZMANN_MEV_PER_K, EX1, example1_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from qudpy_fdgf import (FrequencyPathway, SpectroscopySolver,
                        ThermodynamicContext, standard_nq_protocol)

OUT = FIGURES
DATA = DATA_DIR / "fig_example1_thermal_spectroscopy.npz"

# ------------------------------------------------------------------ model (Example 1, models.py)
n_sites, max_magnons = EX1["n_sites"], EX1["max_magnons"]
temperature, eta = EX1["temperature"], EX1["eta"]                  # K, meV
K_B = BOLTZMANN_MEV_PER_K
model, hamiltonians = example1_model()

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

# ------------------------------------------------------------------ drawing grid
# The figure is drawn on a finer grid than the 60 x 60 grid of the text (checks above and numbers of the
# article); the exact diagonal division makes this a matter of seconds, so it is computed here and not stored.
DRAW_AXES = {"omega_1q": np.linspace(-1.55, -0.55, 181), "omega_emit": np.linspace(0.55, 1.55, 181)}
print(f"drawing grid {DRAW_AXES['omega_1q'].size} x {DRAW_AXES['omega_emit'].size}:", flush=True)
thermal_d = spectrum(ThermodynamicContext(temperature=temperature), axes=DRAW_AXES)
ground_d = spectrum(None, axes=DRAW_AXES)
d1, d3 = DRAW_AXES["omega_1q"], DRAW_AXES["omega_emit"]
total_Td, total_0d = sum(thermal_d.values()), sum(ground_d.values())
hot_d = total_Td - P_0 * total_0d
UNIT = eta ** 2              # amplitudes are drawn in units of 1 / eta^2: the bleach and the emission of the ground state are each worth 1 at the pole

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
fig = plt.figure(figsize=(10.5, 9.6), constrained_layout=True)
grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.8])


def label(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.85))


def displayed(values):
    """Display convention of the article, detection_phase = pi / 2: the real part of i S (GSB and SE positive, ESA negative)."""
    return (1j * values).real


def map_panel(ax, values, title, colorbar_label, signed=False, guides=False):
    """Map of |S| (magma) or of Re(iS) (diverging, symmetric), in units of 1 / eta^2, with the antidiagonal
    omega_emit = -omega_1Q (white or dark dashed line, as in the maps of Example 2)."""
    values = values * UNIT
    if signed:
        vmax = np.abs(values).max()
        image = ax.contourf(d3, d1, values, levels=np.linspace(-vmax, vmax, 31), cmap="RdBu_r")
        line = "0.25"
    else:
        image = ax.contourf(d3, d1, np.abs(values), levels=30, cmap="magma")
        line = "w"
    colorbar = fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.07, pad=0.03)
    colorbar.locator = MaxNLocator(nbins=6)
    colorbar.update_ticks()
    ax.plot(d3, -d3, color=line, ls="--", lw=0.7)
    if guides:
        for x in (bimagnon, bright):
            ax.axvline(x, color=line, ls=":", lw=0.8)
        for y in (-bimagnon, -bright):
            ax.axhline(y, color=line, ls=":", lw=0.8)
    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Emission energy $\omega_{\mathrm{emit}}$ (meV)")
    ax.set_ylabel(r"Excitation energy $\omega_{1Q}$ (meV)")


panels = [("Thermal part, $N\\geq1$", hot_d, r"$|S-P_0S_{0\,\mathrm{K}}|$", r"$\mathrm{Re}[i(S-P_0S_{0\,\mathrm{K}})]$"),
          ("Bimagnon ESA", thermal_d["ESA"], r"$|R_{\mathrm{ESA}}|$", r"$\mathrm{Re}(iR_{\mathrm{ESA}})$"),
          ("Total response, 4 K", total_Td, r"$|S_{\mathrm{R}}|$", r"$\mathrm{Re}(iS_{\mathrm{R}})$")]
axes = []
for column, (title, values, modulus_label, real_label) in enumerate(panels):
    top, bottom = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
    map_panel(top, values, title, modulus_label, guides=(column == 0))
    map_panel(bottom, displayed(values), None, real_label, signed=True, guides=(column == 0))
    axes += [top, bottom]
ax = fig.add_subplot(grid[2, :])

cut = np.argmin(np.abs(d1 + bright))
ax.plot(d3, UNIT * displayed(thermal_d["GSB"][cut] + thermal_d["SE"][cut]), label="GSB + SE")
ax.plot(d3, UNIT * displayed(thermal_d["ESA"][cut]), label="ESA")
ax.plot(d3, UNIT * displayed(total_Td[cut]), "k", lw=2, label=f"total, {temperature:.0f} K")
ax.plot(d3, UNIT * displayed(total_0d[cut]), "k--", lw=1, label="total, 0 K")
ax.axhline(0.0, color="0.6", lw=0.6)
ax.axvline(bright, color="0.4", ls="--", lw=0.8)
ax.axvline(bimagnon, color="C3", ls=":", lw=0.8)
ax.set_xlim(d3[0], d3[-1])
ax.set_title(rf"Emission cut at $\omega_{{1Q}}={d1[cut]:.3f}$ meV")
ax.set_xlabel(r"Emission energy $\omega_{\mathrm{emit}}$ (meV)")
ax.set_ylabel(r"$\mathrm{Re}(iS_{\mathrm{R}})$ ($\eta^{-2}$)")
ax.legend(fontsize=9, loc="upper right")
axes.append(ax)
# lettering by rows: (a)-(c) moduli, (d)-(f) real parts, (g) cut
for ax, text in zip([axes[0], axes[2], axes[4], axes[1], axes[3], axes[5], axes[6]], "abcdefg"):
    label(ax, f"({text})")

print(f"cut at {d1[cut]:.4f} meV: GSB+SE max {UNIT * displayed(thermal_d['GSB'][cut] + thermal_d['SE'][cut]).max():.3f}, "
      f"ESA min {UNIT * displayed(thermal_d['ESA'][cut]).min():.3f} (eta^-2)")
fig.savefig(OUT / "fig_example1_thermal_spectroscopy.pdf")
fig.savefig(OUT / "fig_example1_thermal_spectroscopy.png", dpi=200)
print("saved", OUT / "fig_example1_thermal_spectroscopy.pdf")
