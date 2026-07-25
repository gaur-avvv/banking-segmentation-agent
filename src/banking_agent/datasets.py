from __future__ import annotations

import os
import re
from pathlib import Path


def _candidate_paths(requested: str | None = None) -> list[Path]:
    candidates = []
    if requested:
        raw = str(requested).strip().strip('"')
        candidates.append(Path(raw).expanduser())
        # Accept a Windows path when the CLI is running inside WSL.
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            candidates.append(Path("/mnt", match.group(1).lower(), *re.split(r"[\\/]", match.group(2))))
    configured = os.environ.get("BANKING_DATA_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        Path.cwd() / "data" / "demo",
        Path.cwd() / "data",
        repo_root / "data" / "demo",
        repo_root / "data",
    ])
    return candidates


def resolve_dataset_path(requested: str | None = None) -> tuple[str, str]:
    """Resolve a user path, configured path, or safe clone-local demo dataset."""
    for path in _candidate_paths(requested):
        if path.exists():
            return str(path), "requested/configured dataset"
    from .demo import ensure_demo_data
    fallback = Path.cwd() / "data" / "demo"
    ensure_demo_data(fallback)
    return str(fallback), "no dataset path found; generated safe local demo dataset"


def discover_dataset_paths(base: str | None = None) -> list[dict[str, str]]:
    """Return selectable local dataset paths for the web UI."""
    root = Path(base or Path.cwd()).expanduser()
    paths: list[Path] = []
    for candidate in _candidate_paths(None):
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    if root.exists() and root.is_dir():
        paths.extend(path for path in sorted(root.glob("*.csv")) if path not in paths)
        paths.extend(path for path in sorted(root.glob("*.zip")) if path not in paths)
    return [{"name": path.name, "path": str(path)} for path in paths]
