"""Profile-tagged entry point for the vendored Hermes gateway."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> int:
    """Validate the command-line profile marker, then hand off to Hermes."""

    if (
        len(sys.argv) < 5
        or sys.argv[1] != "hermes_cli.main"
        or not sys.argv[2].startswith("HERMES_HOME=")
    ):
        return 2
    declared = sys.argv[2].partition("=")[2]
    configured = os.environ.get("HERMES_HOME", "")
    try:
        if Path(declared).resolve() != Path(configured).resolve():
            return 2
    except OSError:
        return 2

    # Keep the marker in the operating-system command line so Hermes' process
    # scanner can distinguish the two custom homes, but hide it from its parser.
    del sys.argv[1:3]
    from hermes_cli.main import main as hermes_main

    result = hermes_main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
