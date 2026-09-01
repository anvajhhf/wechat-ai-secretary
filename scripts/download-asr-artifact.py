"""Bounded, resumable HTTPS downloader for the pinned ASR preparation script.

This is a setup tool, never imported by the gateway. Existing completed files
are never overwritten. Partial bytes are usable only after a full hash check.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import stat
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path


def reject_redirected_path(path: Path, project: Path) -> None:
    """Check link objects themselves, including dangling Windows reparse points."""
    if not path.is_relative_to(project):
        raise SystemExit("ASR artifact path left the project")
    for item in (path, *path.parents):
        if item == project:
            break
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise SystemExit("ASR artifact path contains a link/reparse point")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    allowed = project / "runtime" / "models" / "speech"
    target = Path(os.path.abspath(args.target))
    if not target.is_relative_to(allowed):
        raise SystemExit("Unexpected ASR artifact target")
    relative = target.relative_to(allowed)
    if (len(relative.parts) != 2 or relative.parts[0] not in {"sensevoice", "paraformer"}
            or target.name not in {"model.int8.onnx", "tokens.txt"}):
        raise SystemExit("Unexpected ASR artifact target")
    if not re.fullmatch(
        r"https://(?:huggingface\.co|hf-mirror\.com)/csukuangfj/"
        r"sherpa-onnx-[a-z0-9-]+/resolve/[0-9a-f]{40}/(?:model\.int8\.onnx|tokens\.txt)", args.url
    ) or not args.url.endswith("/" + target.name) or not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise SystemExit("Unexpected ASR artifact source")
    if not 1 <= args.size <= 250_000_000:
        raise SystemExit("Unexpected ASR artifact size")
    reject_redirected_path(target, project)
    partial = target.with_name(target.name + ".partial")
    reject_redirected_path(partial, project)
    lock = target.with_name(target.name + ".download.lock")
    reject_redirected_path(lock, project)
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_redirected_path(target, project)
    if target.exists():
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size == args.size and digest == args.sha256:
            print("Verified existing model", flush=True)
            return
        raise SystemExit("Existing model failed verification; will not overwrite")
    if partial.exists() and (not partial.is_file() or partial.stat().st_size > args.size):
        raise SystemExit("Unexpected partial model")
    start = partial.stat().st_size if partial.exists() else 0
    block = 2 * 1024 * 1024
    ranges = [(offset, min(offset + block, args.size) - 1) for offset in range(start, args.size, block)]

    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urlsplit(newurl)
            host = parsed.hostname or ""
            if (parsed.scheme != "https" or parsed.username or parsed.password
                    or parsed.port not in (None, 443)
                    or not any(host == suffix or host.endswith("." + suffix) for suffix in (
                        "huggingface.co", "hf-mirror.com", "hf.co",
                    ))):
                raise ValueError("unexpected-model-redirect")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    def fetch(bounds: tuple[int, int]) -> bytes:
        low, high = bounds
        expected = high - low + 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(args.url, headers={
                    "Range": f"bytes={low}-{high}",
                    "User-Agent": "wechat-secretary-local-model-setup/1",
                })
                opener = urllib.request.build_opener(SafeRedirect())
                with opener.open(req, timeout=35) as response:
                    if not response.url.startswith("https://"):
                        raise ValueError("non-HTTPS redirect")
                    full_small_file = response.status == 200 and low == 0 and expected == args.size
                    valid_range = response.status == 206 and response.headers.get("Content-Range") == f"bytes {low}-{high}/{args.size}"
                    if not (full_small_file or valid_range):
                        raise ValueError("range response mismatch")
                    data = response.read(expected + 1)
                    if len(data) != expected:
                        raise ValueError("range length mismatch")
                    return data
            except Exception:
                if attempt == 2:
                    raise RuntimeError(f"ASR segment {low}-{high} could not be downloaded") from None
                time.sleep(1)
        raise AssertionError("unreachable")

    try:
        lock_file = lock.open("x")
    except FileExistsError:
        raise SystemExit("ASR preparation is already locked; do not run two downloads for one artifact") from None
    try:
        with lock_file:
            # Recheck after acquiring the per-artifact lock. A second setup
            # must not append ranges calculated from an older file length.
            reject_redirected_path(partial, project)
            actual_start = partial.stat().st_size if partial.exists() else 0
            if actual_start != start or target.exists():
                raise SystemExit("Artifact changed during preparation; retry after the other setup ends")
            # Only four bounded chunks in flight; append in byte order. Failed
            # ranges remain resumable and never become a usable model file.
            with partial.open("ab") as output, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for base in range(0, len(ranges), 4):
                    futures = [pool.submit(fetch, bounds) for bounds in ranges[base:base + 4]]
                    for future in futures:
                        output.write(future.result())
                        output.flush()
                        print(f"{target.parent.name}/{target.name}: {output.tell()}/{args.size} bytes", flush=True)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if partial.stat().st_size != args.size or digest != args.sha256:
                raise SystemExit("Artifact hash mismatch; partial file will not be loaded")
            # This setup runs on Windows: rename fails if a final file exists.
            reject_redirected_path(target, project)
            if target.exists():
                raise SystemExit("Artifact appeared during preparation; refusing to replace")
            partial.rename(target)
    finally:
        lock.unlink()
    print(f"Verified {target.parent.name} SHA256 {digest}", flush=True)


if __name__ == "__main__":
    main()
