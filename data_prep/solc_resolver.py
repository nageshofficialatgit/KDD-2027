#!/usr/bin/env python3
"""
Solc Binary Resolver
--------------------
Self-contained module to locate and manage solc compiler binaries.
NEVER calls `solc-select use` — only resolves absolute paths to pre-installed
binaries under ~/.solc-select/artifacts/.

Thread/process safe: no global state mutation.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical version map: partial pragma → full downloadable version
# ---------------------------------------------------------------------------
_PARTIAL_VERSION_MAP: dict[str, str] = {
    "0.4": "0.4.26",
    "0.5": "0.5.17",
    "0.6": "0.6.12",
    "0.7": "0.7.6",
    "0.8": "0.8.20",
}

# Fallback version when pragma is a range that spans multiple majors
_RANGE_FALLBACK: dict[str, str] = {
    "0.4": "0.4.26",
    "0.5": "0.5.17",
    "0.6": "0.6.12",
    "0.7": "0.7.6",
    "0.8": "0.8.20",
}

_ARTIFACTS_DIR = Path.home() / ".solc-select" / "artifacts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_binary(version: str) -> Optional[str]:
    """Return absolute path to the solc binary for *version*, or None."""
    candidate_dir = _ARTIFACTS_DIR / f"solc-{version}"
    # Primary pattern: solc-<version>/solc-<version>
    primary = candidate_dir / f"solc-{version}"
    if primary.exists() and os.access(str(primary), os.X_OK):
        return str(primary)
    # Fallback: solc-<version>/solc
    fallback = candidate_dir / "solc"
    if fallback.exists() and os.access(str(fallback), os.X_OK):
        return str(fallback)
    return None


def install_if_missing(version: str) -> bool:
    """Install *version* via solc-select if not already present. Returns True on success."""
    if resolve_binary(version) is not None:
        return True
    try:
        ret = subprocess.run(
            ["solc-select", "install", version],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return ret.returncode == 0
    except Exception:
        return False


def preflight_install(versions: list[str]) -> dict[str, str]:
    """
    Batch-install all *versions* and return a mapping {version: binary_path}
    for successfully resolved versions.  Call this ONCE before spawning workers.
    """
    solc_map: dict[str, str] = {}
    for v in sorted(set(versions)):
        install_if_missing(v)
        path = resolve_binary(v)
        if path is not None:
            solc_map[v] = path
    return solc_map


def get_all_installed() -> dict[str, str]:
    """Return {version: binary_path} for every version under artifacts/."""
    result: dict[str, str] = {}
    if not _ARTIFACTS_DIR.exists():
        return result
    for entry in sorted(_ARTIFACTS_DIR.iterdir()):
        if entry.is_dir() and entry.name.startswith("solc-"):
            version = entry.name[len("solc-"):]
            path = resolve_binary(version)
            if path is not None:
                result[version] = path
    return result


# ---------------------------------------------------------------------------
# Pragma → Version Detection (standalone, no legacy dependency)
# ---------------------------------------------------------------------------

# Regex: captures the first version-like string from a pragma line
_PRAGMA_RE = re.compile(
    r'pragma\s+solidity\s*[^\d]*((\d+)\.(\d+)(?:\.(\d+))?)',
    re.MULTILINE,
)


def detect_pragma_version(source: str) -> Optional[str]:
    """
    Extract the first Solidity version from source code pragma.
    Returns a full X.Y.Z version string, or None.
    """
    m = _PRAGMA_RE.search(source)
    if m is None:
        return None
    raw = m.group(1)
    major_minor = f"{m.group(2)}.{m.group(3)}"
    if m.group(4) is None:
        # Partial version like "0.8" → map to canonical
        return _PARTIAL_VERSION_MAP.get(major_minor, f"{major_minor}.0")
    return m.group(1).strip()


def version_to_bucket(version: str) -> str:
    """Map a full version string to its major.minor bucket key, e.g. '0.8.20' → '0.8'."""
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version
