"""Regenerate fig_example3_2q_map.pdf (Example 3, fifth-order double-quantum map).

Fifth-order 2Q spectrum of the two-mode model, N_max = 3, 151 x 151 grid, t1 = t3 = t4 = 2 eV^-1.
(a) eta = 8 meV (the calculation of examples/example3_chi5_2q.ipynb, read from
    results/data/example3.npz);
(b) the same calculation with eta = 2 meV (results/data/example3_extra_cases.npz, computed by
    cases.py on first use).
Each panel is normalized to its own maximum, quoted in the panel. White ticks on the vertical axis
mark the three two-excitation energies -E_k. Colored ticks on the horizontal axis mark the three
families of emission resonances of a q = +1 final coherence: eps_j (|1exc><g|), E_k - eps_j
(|2exc><1exc|) and E^(3)_m - E_k (|3exc><2exc|). The circled feature in (a) is the dominant peak
(-E_1, E_1 - eps_1).

Usage:  python fig_example3_2q_map.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (models.py, cases.py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cases import example3_extra_cases
from models import DATA, FIGURES, results_dirs

results_dirs()
saved = np.load(DATA / "example3.npz")
extra = example3_extra_cases()
omega_2q, omega_emit = saved["omega_2q"], saved["omega_emit"]
energies = saved["energies"]
epsilon, E_two, E_three = energies[1:3], energies[3:6], energies[6:10]

panels = (
    (r"(a) $\eta=8$ meV", np.abs(saved["spectrum"])),
    (r"(b) $\eta=2$ meV", np.abs(extra["spectrum_n3_eta2"])),
)
families = (
    (epsilon, "#1fb5d4", r"$\varepsilon_j$"),
    ((E_two[:, None] - epsilon[None, :]).ravel(), "#ff7f0e", r"$E_k-\varepsilon_j$"),
    ((E_three[:, None] - E_two[None, :]).ravel(), "#2ca02c", r"$E^{(3)}_m-E_k$"),
)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
fig = plt.figure(figsize=(6.6, 2.7))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035], left=0.075, right=0.93, bottom=0.30, top=0.90, wspace=0.06)
extent = (omega_emit[0], omega_emit[-1], omega_2q[0], omega_2q[-1])

for index, (title, magnitude) in enumerate(panels):
    ax = fig.add_subplot(grid[0, index])
    image = ax.imshow(magnitude / magnitude.max(), origin="lower", extent=extent, aspect="auto",
                      cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel(r"$\omega_{\rm emit}$ (eV)")
    ax.set_xticks([1.45, 1.50, 1.55, 1.60, 1.65])
    ax.text(0.03, 0.95, rf"$|S|_{{\max}}={magnitude.max() / 10 ** int(np.floor(np.log10(magnitude.max()))):.2f}"
            rf"\times10^{{{int(np.floor(np.log10(magnitude.max())))}}}$",
            transform=ax.transAxes, color="white", va="top")
    ax.hlines(-E_two, 0, 0.04, transform=ax.get_yaxis_transform(), color="white", lw=1.6, clip_on=False)
    for values, color, _ in families:
        inside = values[(values >= omega_emit[0]) & (values <= omega_emit[-1])]
        ax.vlines(inside, 0, 0.07, transform=ax.get_xaxis_transform(), color=color, lw=1.4)
    if index == 0:
        ax.set_ylabel(r"$\omega_{2Q}$ (eV)")
        ax.set_yticks([-3.00, -3.04, -3.08, -3.12, -3.16])
        peak = (E_two[0] - epsilon[0], -E_two[0])
        ax.plot(*peak, "o", mfc="none", mec="white", ms=7, mew=1.2)
        ax.annotate(r"$(-E_1,\,E_1-\varepsilon_1)$", peak, xytext=(peak[0] + 0.012, peak[1] + 0.012),
                    color="white", fontsize=7)
    else:
        ax.set_yticks([-3.00, -3.04, -3.08, -3.12, -3.16])
        ax.tick_params(labelleft=False)

cax = fig.add_subplot(grid[0, 2])
fig.colorbar(image, cax=cax)
cax.set_title(r"$|S|$", fontsize=8)

handles = [Line2D([0], [0], color=color, lw=2) for _, color, _ in families]
fig.legend(handles, [label for *_, label in families], loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.01), handlelength=1.0, columnspacing=1.5)

fig.savefig(FIGURES / "fig_example3_2q_map.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig_example3_2q_map.png", dpi=200, bbox_inches="tight")
print("saved", FIGURES / "fig_example3_2q_map.pdf")
print(f"maxima: eta=8 meV {panels[0][1].max():.3e}, eta=2 meV {panels[1][1].max():.3e}, "
      f"ratio {panels[1][1].max() / panels[0][1].max():.1f}")
