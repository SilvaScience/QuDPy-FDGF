"""The reference test run of the README: examples/old_scripts/quickstart.py, the Quick start of the API manual.

The script prints the position of the linear peak, the width of the absorption line and the rephasing peak of an open
two-level system, each followed by its analytical value. The test runs it as a user does (a fresh interpreter, from the
repository root) and checks that every printed number agrees with the analytical value.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "old_scripts" / "quickstart.py"

GAMMA_1, GAMMA_PHI, ETA, MU = 0.010, 0.005, 1e-4, 0.5      # parameters of the script (eV)
GAMMA_2 = GAMMA_1 / 2 + GAMMA_PHI


def test_quickstart_script_reproduces_the_reference_output():
    assert SCRIPT.is_file(), f"the reference test run of the README needs {SCRIPT}"
    run = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO, capture_output=True, text=True, timeout=300, check=True)
    text = run.stdout

    def numbers(label):
        line = next(l for l in text.splitlines() if l.startswith(label))
        return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", line.split(":", 1)[1].replace("(omega1q, omega3) =", ""))]

    assert abs(numbers("linear peak")[0] - 2.0) < 1e-4
    assert abs(numbers("FWHM (numerical)")[0] - 2 * (GAMMA_2 + ETA)) < 1e-4          # printed with four decimals
    assert abs(numbers("2(Gamma_2 + eta)")[0] - 2 * (GAMMA_2 + ETA)) < 1e-4
    omega_1q, omega_3 = numbers("rephasing peak")
    assert abs(omega_1q + 2.0) < 1e-3 and abs(omega_3 - 2.0) < 1e-3
    expected = 2 * MU**4 / (GAMMA_2 + ETA) ** 2                                      # two pathways, each |mu|^4/(Gamma_2+eta)^2
    assert abs(numbers("|S| at peak")[0] - expected) < 0.01
    assert abs(numbers("2 mu^4/(Gamma_2 + eta)^2")[0] - expected) < 0.01
