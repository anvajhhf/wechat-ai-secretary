"""Profile-tagged entry point for the vendored Hermes gateway."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _project_root() -> Path | None:
    """Return the validated project root that owns the required plugin."""

    root = Path(__file__).resolve().parents[2]
    manifest = root / ".hermes" / "plugins" / "wechat-secretary" / "plugin.yaml"
    return root if manifest.is_file() else None


def main() -> int:
    """Validate the profile and project plugin, then hand off to Hermes."""

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

    # Hermes discovers project plugins relative to the process working
    # directory.  Its Windows detacher normally anchors the child at
    # HERMES_HOME, which would silently bypass this project's fail-closed
    # Weixin hook and let the generic agent answer instead.  Refuse to start
    # without the local plugin and always anchor discovery at this project.
    project_root = _project_root()
    if project_root is None or os.getenv("HERMES_ENABLE_PROJECT_PLUGINS", "").casefold() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return 2
    try:
        os.chdir(project_root)
    except OSError:
        return 2

    # The secretary owns per-message safety/ordering. Its adapter must not
    # silently merge messages or drop fresh replies that repeat earlier text.
    # The compatibility patch leaves ordinary Hermes launches unchanged.
    try:
        from gateway.platforms.weixin import WECHAT_SECRETARY_STRICT_INGRESS_SUPPORTED
    except ImportError:
        return 2
    if WECHAT_SECRETARY_STRICT_INGRESS_SUPPORTED is not True:
        return 2
    os.environ["WECHAT_SECRETARY_STRICT_INGRESS"] = "1"

    # Keep the marker in the operating-system command line so Hermes' process
    # scanner can distinguish the two custom homes, but hide it from its parser.
    del sys.argv[1:3]
    from hermes_cli.main import main as hermes_main

    result = hermes_main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
