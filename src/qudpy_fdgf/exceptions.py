"""Public exception hierarchy for QuDPy-FDGF.

Each exception identifies a distinct failure boundary: invalid external model
contracts, unsupported capabilities, invalid sectors, or
failed iterative convergence. Callers can catch ``QuDPyFDGFError`` for a
single package-level error boundary.
"""


class QuDPyFDGFError(Exception):
    """Base exception raised by QuDPy-FDGF."""


class ModelContractError(QuDPyFDGFError):
    """Raised when an external model violates the expected contract."""


class CapabilityError(QuDPyFDGFError):
    """Raised when a requested capability is unsupported."""


class SectorError(QuDPyFDGFError):
    """Raised for an invalid sector or sector transition."""


class ConvergenceError(QuDPyFDGFError):
    """Raised when an iterative algorithm misses its target tolerance."""


class ModelConsistencyWarning(UserWarning):
    """Emitted when the supplied operators are mutually inconsistent.

    For example, a lowering operator that is not the adjoint of the raising operator, or
    transition elements that the adapter had to discard.
    """


class NullPathwayWarning(UserWarning):
    """Emitted when a pathway written by the user gives an identically zero response.

    Typical causes: an interaction sequence that the transition blocks of the model do not
    allow, or a final coherence order that the detection operator cannot read.
    """


class SteadyStateWarning(UserWarning):
    """Emitted when the stationary state of the generator is not well determined.

    The kernel of ``L`` is not one-dimensional (a closed model, several conserved quantities,
    decoupled sectors without relaxation), or the relaxation gap is too small for the solve to
    resolve it, or the state returned is not a density matrix to the tolerance.
    """


class StationarityError(ModelContractError):
    """Raised when the reference state is not stationary and ``check_stationarity="error"``."""


class StationarityWarning(UserWarning):
    """Emitted when the reference state is not stationary under the generator.

    The perturbative response assumes ``L rho_ref = 0``; a thermal state that does not satisfy
    the detailed balance of the declared collapse channels is not stationary.
    """
