from __future__ import annotations

from pathlib import Path


def list_parquet_inputs(infer_path: str) -> list[str]:
    """Return parquet file URIs/paths under ``infer_path`` (S3 prefix, local dir, or single file)."""
    infer_path = str(infer_path).strip()
    if infer_path.endswith(".parquet"):
        p = Path(infer_path)
        if not infer_path.startswith("s3://"):
            if not p.is_file():
                raise FileNotFoundError(f"Parquet file not found: {infer_path!r}")
            return [str(p.resolve())]
        return [infer_path]

    if infer_path.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem()
        no_scheme = infer_path.replace("s3://", "").rstrip("/")
        if not fs.exists(no_scheme):
            raise FileNotFoundError(f"S3 path does not exist: {infer_path!r}")
        found = fs.find(no_scheme)

        def _is_leaf_parquet(key: str) -> bool:
            base = key.rsplit("/", 1)[-1]
            if not base or base.startswith("_") or base.startswith("."):
                return False
            return base.endswith(".parquet") or "parquet" in base or base.startswith("part-")

        file_uris = sorted({f"s3://{k}" for k in found if _is_leaf_parquet(k)})
        if file_uris:
            return file_uris
        top = fs.ls(no_scheme)
        print(f"[list_parquet_inputs] No leaf parquet; sharding by {len(top)} top-level prefix(es)")
        return sorted({f"s3://{k}" for k in top})

    root = Path(infer_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {infer_path!r}")
    if root.is_file():
        return [str(root)]
    files = sorted(str(p) for p in root.rglob("*.parquet") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .parquet files under {infer_path!r}")
    return files
