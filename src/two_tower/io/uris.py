from __future__ import annotations

from pathlib import Path


def read_uri_bytes(uri: str) -> bytes:
    uri = str(uri)
    if uri.startswith("s3://"):
        import s3fs

        with s3fs.S3FileSystem().open(uri, "rb") as f:
            return f.read()
    return Path(uri).expanduser().resolve().read_bytes()
