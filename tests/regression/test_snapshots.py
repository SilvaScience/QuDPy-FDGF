"""Non-regression of the numerical core against the stored snapshots (see regression_cases.py)."""
from pathlib import Path

import numpy as np
import pytest

from regression_cases import CASES

SNAPSHOTS = Path(__file__).resolve().parent / "data" / "snapshots.npz"
TOLERANCE = 1e-9          # relative to the largest entry of each array


@pytest.fixture(scope="module")
def stored():
    assert SNAPSHOTS.exists(), "run python tests/regression/generate_snapshots.py"
    with np.load(SNAPSHOTS) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.mark.parametrize("case", sorted(CASES))
def test_case_reproduces_its_snapshot(case, stored):
    expected = {key.split("::", 1)[1]: values for key, values in stored.items()
                if key.startswith(case + "::")}
    assert expected, f"no stored array for {case}"
    result = CASES[case]()
    assert set(result) == set(expected), "the set of arrays changed"
    for key, values in result.items():
        reference = expected[key]
        assert values.shape == reference.shape, key
        scale = max(np.abs(reference).max(), 1e-12)
        assert np.abs(values - reference).max() <= TOLERANCE * scale, key
