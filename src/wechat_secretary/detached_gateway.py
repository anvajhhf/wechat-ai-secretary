"""Launch the vendored Hermes gateway through its canonical Windows detacher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Start one hidden gateway process without exposing environment details."""

    if os.name != "nt" or not os.environ.get("HERMES_HOME"):
        print("detached gateway launch is unavailable", file=sys.stderr)
        return 2
    home = str(Path(os.environ["HERMES_HOME"]).resolve())
    os.environ["HERMES_GATEWAY_DETACHED"] = "1"
    try:
        from hermes_cli._subprocess_compat import (
            _WINDOWS_GATEWAY_BREAKAWAY_ENV,
            windows_detach_flags,
            windows_detach_flags_without_breakaway,
        )
        from hermes_cli.gateway_windows import _build_gateway_argv

        argv, working_dir, env_overlay = _build_gateway_argv()
        if len(argv) < 4 or argv[1:3] != ["-m", "hermes_cli.main"]:
            raise RuntimeError("unexpected Hermes launcher shape")
        tagged_argv = [
            argv[0],
            "-m",
            "wechat_secretary.hermes_gateway_entry",
            "hermes_cli.main",
            f"HERMES_HOME={home}",
            *argv[3:],
        ]
        base_env = {**os.environ, **env_overlay}
        log_dir = Path(home) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdio_path = log_dir / "gateway-stdio.log"
        try:
            with stdio_path.open("ab", buffering=0) as log_file:
                process = subprocess.Popen(
                    tagged_argv,
                    cwd=working_dir,
                    env={**base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "1"},
                    creationflags=windows_detach_flags(),
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=log_file,
                )
        except OSError:
            with stdio_path.open("ab", buffering=0) as log_file:
                process = subprocess.Popen(
                    tagged_argv,
                    cwd=working_dir,
                    env={**base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "0"},
                    creationflags=windows_detach_flags_without_breakaway(),
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=log_file,
                )
        pid = process.pid
    except Exception:
        print("detached gateway launch failed", file=sys.stderr)
        return 1
    print(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
