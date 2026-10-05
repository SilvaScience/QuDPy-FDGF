"""Regenerate fig_example1_projection.pdf (Example 1: the readout interaction selects the population).

All nine maps come from a single third-order propagation of the rephasing pathways R1 + R2 + R3 at
t2 = 10 eV^-1 and share one color scale.
Left: the reference chi^(3) response, contracted with the transition operator on the final
one-quantum coherence, without projection.
Right: the four labels accepted by ``fourth_interaction`` (Bu, Kd, Ku, Bd), each combined with the
excited- and ground-state population detectors. Only the two labels with Delta q = -1 (Bu, Kd)
close the q = +1 coherence onto a population, on opposite branches of the density operator; the
others vanish identically. The equality of the non-zero panels with the reference is the content
of S^(4)_Pe = -i S^(3)_pol.

Model, pathways and protocol come from models.py; grid 241 x 241, eta = 0.002 eV.

Usage:  python fig_example1_projection.py
"""
import os
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qudpy_fdgf import ObservableSpec, SpectroscopySolver
from models import EX1, EX1_PATHWAYS, EX1_PROTOCOL, FIGURES, example1_model, results_dirs

results_dirs()
N_POINTS = 241
axes = {"omega_1q": np.linspace(-2.5, -1.5, N_POINTS), "omega_emit": np.linspace(1.5, 2.5, N_POINTS)}
LABELS = (("Bu", -1), ("Kd", -1), ("Ku", +1), ("Bd", +1))
DETECTORS = (("excited_population", r"detect $P_e$"), ("ground_population", r"detect $P_g$"))

# ------------------------------------------------------------------ one propagation, nine maps
solver = SpectroscopySolver(backend="dense", eta=EX1["eta"])
solver.feed_model(example1_model())
observables = {"polarization": "polarization"}
for label, _ in LABELS:
    for operator, _ in DETECTORS:
        observables[f"{label}/{operator}"] = ObservableSpec.action(
            f"{label}/{operator}", fourth_interaction=label, operator=operator)
result = solver.generate_spectrum(EX1_PROTOCOL, axes, pathways=EX1_PATHWAYS,
                                  fixed_coordinates={"t2": EX1["t2"]}, observables=observables)


def total(name):
    return sum(result.observables[name][p.name] for p in EX1_PATHWAYS)


reference = np.abs(total("polarization"))
maps = {key: np.abs(total(key)) for key in observables if key != "polarization"}
vmax = reference.max()
print(f"reference max|S| = {vmax:.4f}")
for key, value in maps.items():
    print(f"  {key:<28} max|S| = {value.max():.4f}")
assert all(np.isclose(v.max(), vmax) or v.max() < 1e-12 for v in maps.values())

# ------------------------------------------------------------------ figure
extent = (axes["omega_emit"][0], axes["omega_emit"][-1], axes["omega_1q"][0], axes["omega_1q"][-1])
cyan = "#1fb5d4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
fig = plt.figure(figsize=(6.6, 2.7))
W, H = 0.135, 0.34
columns = [0.318, 0.468, 0.618, 0.768]
rows = [0.52, 0.12]


def draw(ax, data):
    return ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)


ref_ax = fig.add_axes([0.095, rows[0], W, H])
draw(ref_ax, reference)
ref_ax.set_title(r"reference $\chi^{(3)}$")
ref_ax.set_xlabel(r"$\omega_{\rm emit}$ (eV)")
ref_ax.set_ylabel("detect $\\mu$\n$\\omega_{1Q}$ (eV)")
ref_ax.set_xticks([1.6, 2.0, 2.4])
ref_ax.set_yticks([-2.4, -2.0, -1.6])
ref_ax.text(0.05, 0.93, f"{vmax:.2f}", transform=ref_ax.transAxes, color="white", va="top")
fig.text(0.095 + W / 2, 0.36, r"$S^{(4)}_{P_e}=-i\,S^{(3)}_{\rm pol}$", ha="center", va="top", fontsize=9)
fig.text(0.095 + W / 2, 0.20, "identical modulus", ha="center", va="top", color="0.45", fontsize=7)

image = None
for c, (label, dq) in enumerate(LABELS):
    fig.text(columns[c] + 0.005, rows[0] + H + 0.04, label, fontweight="bold", fontsize=9, ha="left")
    fig.text(columns[c] + W, rows[0] + H + 0.04, rf"$\Delta q={dq:+d}$".replace("+", "+"), fontsize=8, ha="right")
    for r, (operator, row_label) in enumerate(DETECTORS):
        ax = fig.add_axes([columns[c], rows[r], W, H])
        data = maps[f"{label}/{operator}"]
        image = draw(ax, data)
        ax.set_xticks([])
        ax.set_yticks([])
        if data.max() > 1e-12:
            ax.text(0.05, 0.93, f"{data.max():.2f}", transform=ax.transAxes, color="white", va="top")
            for spine in ax.spines.values():
                spine.set_edgecolor(cyan)
                spine.set_linewidth(1.6)
        else:
            ax.text(0.05, 0.93, r"$\equiv 0$", transform=ax.transAxes, color="white", va="top")
        if c == 0:
            ax.set_ylabel(row_label)

colorbar_ax = fig.add_axes([0.925, rows[1], 0.018, rows[0] + H - rows[1]])
fig.colorbar(image, cax=colorbar_ax)
colorbar_ax.set_title(r"$|S|$", fontsize=8)

fig.savefig(FIGURES / "fig_example1_projection.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig_example1_projection.png", dpi=200, bbox_inches="tight")
print("saved", FIGURES / "fig_example1_projection.pdf")
