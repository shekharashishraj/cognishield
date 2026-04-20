from __future__ import annotations

import subprocess
import sys


def test_cli_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "cognishield.app.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "model" in proc.stdout.lower() or "MODEL" in proc.stdout
