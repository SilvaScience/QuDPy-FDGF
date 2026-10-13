"""Liouville pathways and optional sector constraints.

This module represents ordered ket and bra interactions, derives signed
coherence histories, and computes perturbative response prefactors. It also
translates UFSS-style diagram lists into QuDPy-FDGF pathways without requiring
UFSS as a runtime dependency.
"""

from dataclasses import dataclass
import itertools


_COHERENCE_CHANGE = {"Ku": 1, "Kd": -1, "Bu": -1, "Bd": 1}
_KET_LABELS = {"Ku", "Kd"}
_BRA_LABELS = {"Bu", "Bd"}
# Wave-vector sign s_j of pulse j for each label (s_j = Delta q_j) and the labels that carry it.
_LABELS_BY_SIGN = {1: ("Ku", "Bd"), -1: ("Kd", "Bu")}
# k = -k1 + k2 + k3 (rephasing) and k = +k1 - k2 + k3 (non-rephasing), third order.
_STANDARD_SIGNATURES = {"rephasing": (-1, 1, 1), "unrephasing": (1, -1, 1)}
_SIGN_SYMBOLS = {"+": 1, "-": -1, "\u2212": -1}


@dataclass(frozen=True)
class Interaction:
    """Interaction label and optional sectors -> constrained transition."""

    label: str
    operator: str = "light_matter"
    pulse_index: int | None = None
    source_sector: object = None
    target_sector: object = None

    def __post_init__(self):
        label = str(self.label)
        if label not in _COHERENCE_CHANGE:
            raise ValueError(
                f"Unknown interaction {label!r}; use Ku, Kd, Bu, or Bd."
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operator", str(self.operator))
        if self.pulse_index is not None:
            object.__setattr__(self, "pulse_index", int(self.pulse_index))

    @property
    def side(self):
        return "ket" if self.label in _KET_LABELS else "bra"

    @property
    def direction(self):
        """Interaction label -> branch-vector direction.

        Bra-side right multiplication acts through the adjoint.
        """
        return "plus" if self.label in {"Ku", "Bu"} else "minus"

    @property
    def sign(self):
        """Wave-vector sign s_j of the pulse for this label (equal to its coherence change)."""
        return _COHERENCE_CHANGE[self.label]

    @property
    def perturbative_prefactor(self):
        """Return this interaction's pathway-prefactor contribution."""
        return 1j * (-1 if self.side == "bra" else 1)


def _label_of(interaction):
    return interaction.label if isinstance(interaction, Interaction) else str(interaction)


def pathway_signature(interactions):
    """Ordered interactions -> signature ``(s_1, ..., s_n)`` of wave-vector signs.

    ``s_j = +1`` for ``Ku`` and ``Bd`` and ``-1`` for ``Kd`` and ``Bu``, so that the signal is
    radiated in the direction ``sum_j s_j k_j``. Rephasing is ``(-1, +1, +1)`` and
    non-rephasing ``(+1, -1, +1)`` at third order.
    """
    return tuple(_COHERENCE_CHANGE[_label_of(item)] for item in interactions)


def coherence_orders_from_interactions(interactions):
    """Ordered interactions -> signed coherence history."""
    current = 0
    history = []
    for interaction in interactions:
        label = (
            interaction.label
            if isinstance(interaction, Interaction)
            else str(interaction)
        )
        current += _COHERENCE_CHANGE[label]
        history.append(current)
    return tuple(history)


@dataclass(frozen=True)
class FrequencyPathway:
    """Generic definition of an ordered pathway."""

    name: str
    interactions: tuple
    component: str = "custom"
    amplitude: complex = 1.0
    prefactor: complex | None = None
    coherence_orders: tuple = ()
    detection: str = "polarization"
    # "user" for a pathway written by hand; "enumerated" and "ufss" for generated lists, whose
    # zero members are expected and are not reported as errors.
    origin: str = "user"

    def __post_init__(self):
        interactions = tuple(
            item
            if isinstance(item, Interaction)
            else Interaction(**item)
            if isinstance(item, dict)
            else Interaction(str(item))
            for item in self.interactions
        )
        if not interactions:
            raise ValueError("A pathway must contain interactions.")
        expected = coherence_orders_from_interactions(interactions)
        coherence_orders = (
            tuple(int(item) for item in self.coherence_orders)
            if self.coherence_orders
            else expected
        )
        if coherence_orders != expected:
            raise ValueError(
                f"Pathway {self.name!r} declares {coherence_orders}, "
                f"but its interactions imply {expected}."
            )
        component = str(self.component).lower().replace("-", "")
        if component in {"nonrephasing", "nonrephase"}:
            component = "unrephasing"
        signature = pathway_signature(interactions)
        standard = _STANDARD_SIGNATURES.get(component)
        if standard is not None and len(signature) == len(standard) and signature != standard:
            raise ValueError(
                f"Pathway {self.name!r} is declared as component {component!r} "
                f"(signature {standard}), but its interactions have signature {signature}. "
                "Summing pathways of different signatures into one component mixes phase-matching "
                "directions; use another component name for a custom grouping."
            )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "coherence_orders", coherence_orders)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "origin", str(self.origin))
        object.__setattr__(self, "amplitude", complex(self.amplitude))
        object.__setattr__(self, "detection", str(self.detection))
        if self.prefactor is not None:
            object.__setattr__(self, "prefactor", complex(self.prefactor))

    @property
    def signature(self):
        """Signature ``(s_1, ..., s_n)`` of wave-vector signs (see ``pathway_signature``)."""
        return pathway_signature(self.interactions)

    @property
    def bra_sign(self):
        count = sum(item.side == "bra" for item in self.interactions)
        return -1 if count % 2 else 1

    @property
    def response_prefactor(self):
        if self.prefactor is not None:
            return self.prefactor
        return (
            complex(self.amplitude)
            * (1j ** len(self.interactions))
            * self.bra_sign
        )

    def metadata(self):
        return {
            "name": self.name,
            "component": self.component,
            "amplitude": self.amplitude,
            "prefactor": self.prefactor,
            "response_prefactor": self.response_prefactor,
            "coherence_orders": self.coherence_orders,
            "signature": self.signature,
            "detection": self.detection,
            "origin": self.origin,
            "interactions": tuple(
                {
                    "label": item.label,
                    "operator": item.operator,
                    "pulse_index": item.pulse_index,
                    "source_sector": item.source_sector,
                    "target_sector": item.target_sector,
                }
                for item in self.interactions
            ),
        }


def translate_ufss_diagrams(
    diagrams,
    *,
    component="custom",
    names=None,
    operator="light_matter",
    amplitudes=None,
    prefactors=None,
    detection="polarization",
):
    """UFSS diagram list -> QuDPy-FDGF pathways without an UFSS dependency."""
    diagrams = list(diagrams)
    size = len(diagrams)
    names = list(names) if names is not None else [None] * size
    amplitudes = (
        list(amplitudes) if amplitudes is not None else [1.0] * size
    )
    prefactors = (
        list(prefactors) if prefactors is not None else [None] * size
    )
    if not all(len(items) == size for items in (names, amplitudes, prefactors)):
        raise ValueError("Metadata entries must match the diagrams.")

    pathways = []
    used_names = set()
    for index, (diagram, name, amplitude, prefactor) in enumerate(
        zip(diagrams, names, amplitudes, prefactors), start=1
    ):
        interactions = tuple(
            Interaction(
                label=item[0],
                pulse_index=item[1],
                operator=operator,
            )
            for item in diagram
        )
        base_name = str(name or f"P{index}")
        unique_name = base_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)
        pathways.append(
            FrequencyPathway(
                name=unique_name,
                interactions=interactions,
                component=component,
                amplitude=amplitude,
                prefactor=prefactor,
                detection=detection,
                origin="ufss",
            )
        )
    return pathways


def _parse_signature(signature):
    if isinstance(signature, str):
        symbols = [char for char in signature if char not in " ,()[]"]
        try:
            values = tuple(_SIGN_SYMBOLS[char] for char in symbols)
        except KeyError as exc:
            raise ValueError(
                f"Unknown sign {exc.args[0]!r} in signature {signature!r}; use '+' and '-'."
            ) from exc
    else:
        values = tuple(int(item) for item in signature)
    if not values or any(item not in (1, -1) for item in values):
        raise ValueError(
            "A signature is a non-empty sequence of +1 and -1 (or a string of '+' and '-')."
        )
    return values


def pathways_from_signature(
    signature,
    *,
    component="custom",
    detection="polarization",
    operator="light_matter",
    max_manifold=None,
    min_manifold=None,
):
    """Signature -> every Liouville pathway with that wave-vector signature.

    Pulse ``j`` with sign ``s_j = +1`` can act as ``Ku`` or ``Bd`` and with ``s_j = -1`` as
    ``Kd`` or ``Bu``: the ``2**n`` ordered label sequences are returned, named after their labels
    (``"Bu-Ku-Bd"``) and marked ``origin="enumerated"``. Those that the model does not allow
    vanish when the spectrum is computed (``SpectrumResult.null_pathways()`` lists them), so the
    set is complete by construction, including the sequences that only a thermal reference state
    or a population transfer makes non-zero.

    ``max_manifold`` and ``min_manifold`` reproduce the rule 4(a) of the UFSS diagram generator:
    ket and bra manifold indices start at 0 and a sequence is dropped when one leaves
    ``[min_manifold, max_manifold]``. This is valid for a ground-state reference state without
    relaxation between manifolds only; with relaxation or thermal populations, leave both
    ``None``.
    """
    signs = _parse_signature(signature)
    low = -float("inf") if min_manifold is None else min_manifold
    high = float("inf") if max_manifold is None else max_manifold
    step = {"Ku": (1, 0), "Kd": (-1, 0), "Bu": (0, 1), "Bd": (0, -1)}
    pathways = []
    for labels in itertools.product(*(_LABELS_BY_SIGN[sign] for sign in signs)):
        ket = bra = 0
        allowed = True
        for label in labels:
            ket += step[label][0]
            bra += step[label][1]
            if not (low <= ket <= high and low <= bra <= high):
                allowed = False
                break
        if allowed:
            pathways.append(
                FrequencyPathway(
                    name="-".join(labels),
                    interactions=tuple(Interaction(label, operator=operator) for label in labels),
                    component=component,
                    detection=detection,
                    origin="enumerated",
                )
            )
    return pathways
