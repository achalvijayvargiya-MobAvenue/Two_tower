"""Helpers to inspect infer Parquet inputs and the user-ranking output contract (no GPU required for peek)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.dataset as pads

from two_tower.configs import InferJobConfig
from two_tower.inference.artifact_paths import load_vocab_artifact_pickle, training_artifact_uris
from two_tower.inference.list_inputs import list_parquet_inputs
from two_tower.inference.worker import _resolve_source_device_id_column


def ranking_output_columns(ranking_device_id_col: str) -> list[str]:
    """Column names written to each user-ranking Parquet part."""
    return [ranking_device_id_col, "client_id", "score", "rank"]


def ranking_output_contract_text(ranking_device_id_col: str) -> str:
    c = ranking_device_id_col
    return (
        "User ranking Parquet schema (one row per user x top-k client):\n"
        f"  - {c}  (Utf8): copied from the *inference input* Parquet row, after choosing a source column; "
        "not from the model.\n"
        "  - client_id  (Utf8): matched client from client_embeddings.\n"
        "  - score  (Float32): dot-product similarity.\n"
        "  - rank  (Int32): 1 .. topk within that user.\n"
        "The source column for the id is resolved per input file from candidates (YAML + vocab + fallbacks); "
        "see peek_infer_device_resolution()."
    )


def build_device_id_candidates(cfg: InferJobConfig, vocab_device_id_col: str) -> list[str]:
    """Same order as ``run_inference_job``."""
    device_id_candidates: list[str] = []
    if cfg.infer_parquet_device_id_col is not None:
        device_id_candidates.append(cfg.infer_parquet_device_id_col)
    device_id_candidates.append(vocab_device_id_col)
    for alt in ("device_id", "dev_id", "ifa", "advertising_id"):
        if alt not in device_id_candidates:
            device_id_candidates.append(alt)
    return device_id_candidates


def peek_infer_device_resolution(
    parquet_path: str | Path,
    device_id_candidates: list[str],
    *,
    peek_batch_rows: int = 65_536,
) -> tuple[str, int, int]:
    """Return ``(source_column_name, non_null_in_peek, peek_row_count)`` using worker logic."""
    p = str(parquet_path)
    dset = pads.dataset(p, format="parquet")
    return _resolve_source_device_id_column(
        dset, p, device_id_candidates, int(peek_batch_rows), worker_id=0, verbose=False
    )


def sample_source_column_values(
    parquet_path: str | Path,
    source_col: str,
    *,
    limit: int = 8,
) -> pl.DataFrame:
    """Read up to ``limit`` non-null values from ``source_col`` (column projection only)."""
    p = str(parquet_path)
    dset = pads.dataset(p, format="parquet")
    if source_col not in dset.schema.names:
        return pl.DataFrame({source_col: []})
    t = pl.read_parquet(p, columns=[source_col])
    return t.filter(pl.col(source_col).is_not_null()).head(limit)


def run_preflight(cfg: InferJobConfig, *, peek_batch_rows: int = 65_536, sample_limit: int = 8) -> dict[str, Any]:
    """Load vocab + first infer file; print-ready summary dict."""
    arts = training_artifact_uris(cfg.paths.artifacts_base)
    vocab = load_vocab_artifact_pickle(arts["vocab"])
    voc_dev = str(vocab["device_id_col"])
    candidates = build_device_id_candidates(cfg, voc_dev)
    out_col = str(cfg.infer.ranking_device_id_col)
    files = list_parquet_inputs(cfg.paths.infer)
    summary: dict[str, Any] = {
        "vocab_device_id_col": voc_dev,
        "ranking_device_id_col": out_col,
        "ranking_output_columns": ranking_output_columns(out_col),
        "device_id_candidates": candidates,
        "infer_files_found": len(files),
        "first_infer_file": files[0] if files else None,
    }
    if not files:
        summary["peek"] = None
        summary["sample"] = None
        return summary
    src, nn, nr = peek_infer_device_resolution(files[0], candidates, peek_batch_rows=peek_batch_rows)
    summary["peek"] = {"source_column": src, "non_null_in_peek": nn, "peek_rows": nr}
    summary["sample"] = sample_source_column_values(files[0], src, limit=sample_limit)
    return summary


def verify_ranking_parquet(path: str | Path, ranking_device_id_col: str) -> dict[str, Any]:
    """Null/blank counts for the device column in a written ranking file."""
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {p}"}
    df = pl.read_parquet(p)
    if ranking_device_id_col not in df.schema:
        return {
            "ok": False,
            "error": f"missing column {ranking_device_id_col!r}; have {df.columns}",
        }
    s = df[ranking_device_id_col].cast(pl.Utf8, strict=False)
    n = len(s)
    nulls = int(s.is_null().sum())
    blanks = int(s.str.strip_chars().eq("").fill_null(False).sum()) if n else 0
    return {
        "ok": nulls == 0 and blanks == 0,
        "rows": n,
        "nulls": nulls,
        "blank_strings": blanks,
        "head": df.select(ranking_device_id_col).head(5),
    }
