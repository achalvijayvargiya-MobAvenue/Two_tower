from __future__ import annotations

from pathlib import Path


def artifact_uri(base: str, *relative_parts: str) -> str:
    """Join S3 or local base with relative path parts."""
    rel = "/".join(relative_parts)
    b = base.rstrip("/")
    if b.startswith("s3://"):
        return f"{b}/{rel}"
    return str(Path(b) / rel)
