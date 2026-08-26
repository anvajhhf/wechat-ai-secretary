"""Stop exactly one secretary profile gateway without a global process scan."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def _same_path(left: object, right: Path) -> bool:
    try:
        return Path(str(left)).resolve() == right.resolve()
    except OSError:
        return False


def _validated_process(home: Path):
    state_path = home / "gateway_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(state["pid"])
        recorded_start = state["start_time"]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if pid <= 0 or not _same_path(state.get("hermes_home"), home):
        return None

    try:
        import psutil

        process = psutil.Process(pid)
        from gateway.status import _get_process_start_time, looks_like_gateway_command_line

        actual_start = _get_process_start_time(pid)
        if actual_start is None or str(actual_start) != str(recorded_start):
            return None

        if not looks_like_gateway_command_line(" ".join(process.cmdline())):
            return None
        return process
    except (OSError, ValueError, ImportError):
        return None
    except Exception as exc:
        # psutil uses platform-specific exception classes; keep this helper
        # fail-closed without echoing process details.
        if exc.__class__.__module__.startswith("psutil"):
            return None
        raise


def _recorded_pid_is_alive(home: Path) -> bool:
    """Fail-closed probe used when full process identity validation fails."""

    try:
        state = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
        pid = int(state["pid"])
        import psutil

        return pid > 0 and psutil.pid_exists(pid)
    except (OSError, ValueError, TypeError, KeyError, ImportError, json.JSONDecodeError):
        return False


def _begin_quiet_maintenance(home: Path) -> bool:
    """Suppress only the redundant home-channel stop notice during maintenance."""

    try:
        from gateway.drain_control import write_drain_request

        write_drain_request(
            principal="wechat-secretary-maintenance",
            suppress_notification=True,
            home=home,
        )
        return True
    except Exception:
        return False


def _end_quiet_maintenance(home: Path) -> None:
    try:
        from gateway.drain_control import clear_drain_request

        clear_drain_request(home=home)
    except Exception:
        pass


def main() -> int:
    configured = os.environ.get("HERMES_HOME", "")
    if not configured:
        print("profile gateway stop unavailable", file=sys.stderr)
        return 2
    home = Path(configured).resolve()
    process = _validated_process(home)
    if process is None:
        if _recorded_pid_is_alive(home):
            print("profile gateway identity mismatch", file=sys.stderr)
            return 1
        print("not-running")
        return 0

    quiet_maintenance = _begin_quiet_maintenance(home)
    try:
        try:
            from gateway.status import write_planned_stop_marker

            write_planned_stop_marker(process.pid)
        except Exception:
            pass

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not process.is_running():
                print("stopped")
                return 0
            try:
                process.wait(timeout=0.5)
                print("stopped")
                return 0
            except Exception as exc:
                if exc.__class__.__module__.startswith("psutil"):
                    continue
                raise

        # Re-read and revalidate the exact PID/start-time pair before escalation.
        process = _validated_process(home)
        if process is None:
            print("stopped")
            return 0
        try:
            process.kill()
            process.wait(timeout=5.0)
        except Exception as exc:
            if not exc.__class__.__module__.startswith("psutil"):
                raise
            if process.is_running():
                print("profile gateway stop failed", file=sys.stderr)
                return 1
        print("stopped")
        return 0
    finally:
        if quiet_maintenance:
            _end_quiet_maintenance(home)


if __name__ == "__main__":
    raise SystemExit(main())
