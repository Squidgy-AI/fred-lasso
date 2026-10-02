"""Configuration, resolved the same way the VAST Builders Challenge skills do.

On the workshop VM everything lives in the single ``/config/<team>.config`` file and is
usually already exported. Locally, nothing is, so every getter falls back to the
environment and then to None, and callers decide whether that is fatal.
"""
from __future__ import annotations

import glob
import os
import shlex
from functools import lru_cache

CONFIG_GLOB = "/config/*.config"

# Fixed taxonomy for the constrained second pass. Order matters: it is the order the
# options are shown to Cosmos, and the index is what we parse back out.
ASSIST_TYPES = [
    "cross from wide",
    "through ball",
    "set piece",
    "long-range shot",
    "rebound",
    "other",
]


@lru_cache(maxsize=1)
def load_team_config() -> dict[str, str]:
    """Parse the team config file into a dict, without exporting it.

    Returns an empty dict off-VM. Values already in the environment win, matching the
    skills' note that WANDB_* are often exported even when absent from the file.
    """
    matches = sorted(glob.glob(CONFIG_GLOB))
    parsed: dict[str, str] = {}
    if len(matches) == 1:
        with open(matches[0], encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                try:
                    value = " ".join(shlex.split(raw))
                except ValueError:
                    value = raw.strip()
                parsed[key.strip()] = value
    return parsed


def get(name: str, default: str | None = None) -> str | None:
    """Environment first, then the team config file, then the default."""
    value = os.environ.get(name)
    if value:
        return value
    value = load_team_config().get(name)
    return value if value else default


def require(name: str) -> str:
    value = get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. On the workshop VM it comes from {CONFIG_GLOB}; "
            f"locally, export it or pass an explicit override."
        )
    return value


def config_source() -> str:
    matches = sorted(glob.glob(CONFIG_GLOB))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return f"ambiguous ({len(matches)} files match {CONFIG_GLOB})"
    return "environment only (no team config file found)"


def describe() -> str:
    """A credential-safe summary, for the health check and the README.

    Prints only whether a value is present, never the value, per the ask-cosmos rules.
    """
    keys = [
        "INGRESS_URL", "USERNAME", "PASSWORD",
        "S3_ENDPOINT", "ACCESS_KEY", "SECRET_KEY",
        "S3_CHUNKS_BUCKET", "S3_SEGMENTS_BUCKET",
        "VDB_ENDPOINT", "VASTDB_BUCKET", "VDB_SCHEMA", "VDB_COLLECTION",
        "COSMOS3_REASON_URL", "COSMOS3_REASON_MODEL",
        "COSMOS_EMBED1_URL", "COSMOS_EMBED1_MODEL",
        "YOLO_URL", "CANARY_1B_URL",
        "WANDB_API_KEY", "WANDB_TEAM", "WANDB_PROJECT",
    ]
    lines = [f"config source: {config_source()}"]
    for key in keys:
        lines.append(f"  {'set  ' if get(key) else 'MISSING'}  {key}")
    return "\n".join(lines)
