from __future__ import annotations

import base64
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    path = Path(sys.argv[1]).resolve(strict=True)
    source = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    snapshot = sqlite3.connect(":memory:")
    try:
        source.backup(snapshot)
        payload = snapshot.serialize()
    finally:
        snapshot.close()
        source.close()
    sys.stdout.buffer.write(base64.b64encode(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
