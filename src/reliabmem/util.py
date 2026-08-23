from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import subprocess
import tempfile


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free estimate used for budgets, never billing."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 3.5))


def pack_chronological(items: Iterable[tuple[int, str]], budget: int) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    used = 0
    for turn_id, text in reversed(list(items)):
        cost = estimate_tokens(text)
        if cost > budget and not selected:
            # Retain the tail of one oversized item rather than violate the budget.
            chars = max(1, int(budget * 3.5))
            selected.append((turn_id, text[-chars:]))
            break
        if used + cost > budget:
            continue
        selected.append((turn_id, text))
        used += cost
    return sorted(selected)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def code_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        files = sorted(p for p in (root / "src").rglob("*.py"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return f"tree-{digest.hexdigest()}"
