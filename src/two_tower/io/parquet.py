from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParquetListing:
    files: list[str]


def list_parquet_uris(prefix_or_file: str) -> ParquetListing:
    """List parquet files under an S3/local prefix, or return the single file.

    Implementation will be added when we wire in s3fs/pyarrow.
    """
    raise NotImplementedError


def read_parquet_pandas(uri: str):
    """Read a parquet dataset/file into a pandas DataFrame."""
    raise NotImplementedError


def iter_record_batches(uri: str, columns: list[str], batch_size: int) -> Iterable[object]:
    """Stream parquet as Arrow RecordBatches (pyarrow.dataset Scanner style)."""
    raise NotImplementedError


def write_parquet_pandas(df, uri: str) -> None:
    """Write DataFrame to parquet (S3/local)."""
    raise NotImplementedError

