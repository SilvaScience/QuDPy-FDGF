"""Result containers independent of plotting tools.

These lightweight data classes store individual pathway responses,
multidimensional spectra, component sums, observables, diagnostics, and plot
artifacts. Keeping them free of plotting dependencies lets numerical results
move cleanly between analysis and visualization workflows.
"""

from dataclasses import dataclass, field


@dataclass
class PathwayResult:
    """Pathway value and its numerical diagnostics."""

    value: complex
    pathway: str
    diagnostics: dict = field(default_factory=dict)
    observables: dict = field(default_factory=dict)


@dataclass
class SpectrumResult:
    """Multidimensional responses grouped by pathway and component."""

    axis_names: tuple
    axis_values: tuple
    pathways: dict
    components: dict
    fixed_coordinates: dict
    pathway_metadata: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    observables: dict = field(default_factory=dict)
    observable_components: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axis_values)

    def null_pathways(self, tolerance=1e-14):
        """Pathways whose response vanishes on the whole grid.

        A pathway is null when ``max |S| <= tolerance * max_p max |S_p|``, the second maximum
        running over the pathways of this result (any pathway of an all-zero result is null).
        """
        peaks = {name: float(abs(values).max()) for name, values in self.pathways.items()}
        scale = max(peaks.values(), default=0.0)
        return tuple(name for name, peak in peaks.items() if peak <= tolerance * scale)


@dataclass
class PlotResult:
    """Artifacts returned by ``SpectroscopyPlotter``."""

    figure: object
    axes: object
    panel_names: tuple
    data: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
