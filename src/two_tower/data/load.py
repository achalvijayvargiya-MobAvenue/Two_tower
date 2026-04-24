from __future__ import annotations

import pandas as pd

from two_tower.configs import PipelineConfig


def _balance_train_per_client(
    df: pd.DataFrame,
    neg_per_pos: int | None,
    *,
    ycol: str,
    ccol: str,
    rs: int,
) -> pd.DataFrame:
    """Undersample within each client so negatives:positives ≈ r:1 (no duplication)."""
    if neg_per_pos is None:
        return df
    r = max(1, int(neg_per_pos))
    parts: list[pd.DataFrame] = []
    for i, (_cid, g) in enumerate(df.groupby(ccol, sort=False)):
        pos = g[g[ycol] == 1]
        neg = g[g[ycol] == 0]
        n_p, n_n = len(pos), len(neg)
        if n_p == 0 or n_n == 0:
            parts.append(g)
            continue
        k = min(n_p, n_n // r)
        rs_p, rs_n = rs + 2 * i, rs + 2 * i + 1
        if k == 0:
            m = min(n_p, n_n)
            parts.append(
                pd.concat(
                    [pos.sample(n=m, random_state=rs_p), neg.sample(n=m, random_state=rs_n)],
                    ignore_index=True,
                )
            )
        else:
            parts.append(
                pd.concat(
                    [pos.sample(n=k, random_state=rs_p), neg.sample(n=k * r, random_state=rs_n)],
                    ignore_index=True,
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _equalize_clients_to_min_rows_keep_ratio(
    df: pd.DataFrame,
    neg_per_pos: int | None,
    *,
    ycol: str,
    ccol: str,
    rs: int,
) -> pd.DataFrame:
    """Downsample each client to the same row count while keeping neg:pos = r:1 per client."""
    if neg_per_pos is None:
        return df
    r = max(1, int(neg_per_pos))
    stats: list[tuple[object, int, int]] = []
    for _cid, g in df.groupby(ccol, sort=False):
        pos_c = int((g[ycol] == 1).sum())
        neg_c = int((g[ycol] == 0).sum())
        stats.append((_cid, pos_c, neg_c))
    if not stats:
        return df
    m = min(min(pos_c, (neg_c // r) if r else pos_c) for _cid, pos_c, neg_c in stats)
    if m <= 0:
        return df
    parts: list[pd.DataFrame] = []
    for i, (_cid, g) in enumerate(df.groupby(ccol, sort=False)):
        pos = g[g[ycol] == 1]
        neg = g[g[ycol] == 0]
        rs_p, rs_n = rs + 3 * i, rs + 3 * i + 1
        parts.append(
            pd.concat(
                [pos.sample(n=m, random_state=rs_p), neg.sample(n=m * r, random_state=rs_n)],
                ignore_index=True,
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _maybe_downsample_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tc = cfg.train
    ycol = cfg.features.label_col
    ccol = cfg.features.client_id_col

    if ccol not in train_df.columns or ccol not in val_df.columns:
        return train_df, val_df

    r = getattr(tc, "downsample_neg_per_pos", None)
    rs = int(getattr(tc, "downsample_random_state", 42))
    eq = bool(getattr(tc, "downsample_equalize_client_rows", True))

    if bool(getattr(tc, "downsample_train", False)):
        before = len(train_df)
        work = _balance_train_per_client(train_df, r, ycol=ycol, ccol=ccol, rs=rs)
        if r is not None and eq:
            work = _equalize_clients_to_min_rows_keep_ratio(work, r, ycol=ycol, ccol=ccol, rs=rs + 1000)
        train_df = work.sample(frac=1.0, random_state=rs).reset_index(drop=True)
        print(
            f"[downsample] train: {before:,} -> {len(train_df):,} "
            f"(neg:pos={r}:1, equalize={eq})"
        )

    if bool(getattr(tc, "downsample_val", False)):
        before = len(val_df)
        work = _balance_train_per_client(val_df, r, ycol=ycol, ccol=ccol, rs=rs + 7)
        if r is not None and eq:
            work = _equalize_clients_to_min_rows_keep_ratio(work, r, ycol=ycol, ccol=ccol, rs=rs + 2000)
        val_df = work.sample(frac=1.0, random_state=rs + 7).reset_index(drop=True)
        print(
            f"[downsample] val: {before:,} -> {len(val_df):,} "
            f"(neg:pos={r}:1, equalize={eq})"
        )

    return train_df, val_df


def merge_client_metadata_into_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-join ``client_*`` columns from client-metadata Parquet onto train/val by ``client_id_col``.

    Expects the same layout as the SQL ``client_metadata`` table (key column ``client_bundle_id``
    unless ``features.client_id_col`` is set to another name present in both frames and metadata).
    """
    dl = cfg.data_load
    if not dl.client_metadata_uri:
        raise ValueError("merge_client_metadata: set data.client_metadata_uri in config")

    key = cfg.features.client_id_col
    if key not in train_df.columns:
        raise KeyError(f"train data missing join column {key!r}")
    if key not in val_df.columns:
        raise KeyError(f"val data missing join column {key!r}")

    cmf = pd.read_parquet(dl.client_metadata_uri)
    if key not in cmf.columns:
        raise KeyError(f"client_metadata missing join column {key!r}; have: {list(cmf.columns)[:40]}")

    need = list(cfg.features.client_feature_cols)
    missing_meta = [c for c in need if c not in cmf.columns]
    if missing_meta:
        raise KeyError(
            "client_metadata missing configured client_feature_cols: "
            f"{missing_meta[:20]}{'...' if len(missing_meta) > 20 else ''}"
        )

    meta = cmf[[key, *need]].drop_duplicates(subset=[key], keep="first")

    def _apply(df: pd.DataFrame, name: str) -> pd.DataFrame:
        out = df.drop(columns=[c for c in need if c in df.columns], errors="ignore")
        merged = out.merge(meta, on=key, how="left", validate="many_to_one")
        bad = merged[need].isna().any(axis=1)
        if bad.any():
            missing_ids = merged.loc[bad, key].drop_duplicates().tolist()
            sample = missing_ids[:15]
            raise ValueError(
                f"{name}: {int(bad.sum())} rows have no client_metadata match for {key!r}; "
                f"example ids: {sample}{'...' if len(missing_ids) > 15 else ''}"
            )
        return merged

    train_m = _apply(train_df, "train")
    val_m = _apply(val_df, "val")
    print(
        f"[merge_client] joined {len(need)} client columns from {dl.client_metadata_uri!r} "
        f"on {key!r}; train={len(train_m):,} val={len(val_m):,}"
    )
    return train_m, val_m


def _resolve_injected_client_id(crow: pd.Series, cfg: PipelineConfig) -> tuple[str | None, object | None]:
    candidates = [
        cfg.features.client_id_col,
        "client_id",
        "client_bundle_id",
        "client_bundle",
        "clientId",
        "bundle_id",
        "client",
    ]
    for col in candidates:
        if col and col in crow.index:
            return col, crow[col]

    row_filter = cfg.data_load.single_client_row_filter or {}
    for col in candidates:
        if col and col in row_filter:
            return cfg.features.client_id_col, row_filter[col]

    return None, None


def load_train_validation_frames(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation Parquet from S3 or local paths (``pd.read_parquet``).

    Optional:

    - **Per-row** client features via ``merge_client_metadata`` + ``client_metadata_uri``
      (join on ``features.client_id_col``).
    - **Single-client** broadcast when ``inject_single_client_metadata`` is True.
    """
    train_df = pd.read_parquet(cfg.paths.train)
    val_df = pd.read_parquet(cfg.paths.val)

    dl = cfg.data_load
    if dl.merge_client_metadata and dl.inject_single_client_metadata:
        raise ValueError(
            "Use either data.merge_client_metadata or data.inject_single_client_metadata, not both."
        )
    if dl.merge_client_metadata:
        train_df, val_df = merge_client_metadata_into_frames(train_df, val_df, cfg)

    if dl.inject_single_client_metadata:
        if dl.single_client_metadata_uri:
            cmf = pd.read_parquet(dl.single_client_metadata_uri)
            if dl.single_client_row_filter:
                for fk, fv in dl.single_client_row_filter.items():
                    if fk not in cmf.columns:
                        raise KeyError(
                            f"metadata missing filter column {fk!r}; "
                            f"have: {list(cmf.columns)[:30]} ..."
                        )
                    cmf = cmf[cmf[fk] == fv]
            if len(cmf) == 0:
                raise ValueError("inject_single_client_metadata: no row left after single_client_row_filter")
            crow = cmf.iloc[0]
        elif dl.single_client_features_hardcoded:
            crow = pd.Series(dl.single_client_features_hardcoded)
        else:
            raise ValueError(
                "inject_single_client_metadata is True: set single_client_metadata_uri or "
                "single_client_features_hardcoded in config data section"
            )
        inj = 0
        for col in cfg.features.client_feature_cols:
            if col not in crow.index:
                continue
            train_df[col] = crow[col]
            val_df[col] = crow[col]
            inj += 1

        injected_id_col, injected_id_val = _resolve_injected_client_id(crow, cfg)
        if injected_id_col is not None:
            train_df[cfg.features.client_id_col] = injected_id_val
            val_df[cfg.features.client_id_col] = injected_id_val
            print(
                f"[inject_client] set client id column {cfg.features.client_id_col!r} "
                f"from metadata field {injected_id_col!r}"
            )
        print(
            f"[inject_client] broadcast {inj} client columns from one metadata row "
            f"onto train={len(train_df):,} val={len(val_df):,} rows"
        )

    if cfg.features.label_col not in train_df.columns:
        raise KeyError(
            f"Train data missing label column {cfg.features.label_col!r}. "
            f"Columns: {list(train_df.columns)[:40]} ..."
        )

    train_df, val_df = _maybe_downsample_frames(train_df, val_df, cfg)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    return train_df, val_df
