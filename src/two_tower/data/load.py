from __future__ import annotations

import pandas as pd

from two_tower.configs import PipelineConfig


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

    Mirrors the reference notebook: optional broadcast of one client-metadata row
    onto all rows when ``cfg.data_load.inject_single_client_metadata`` is True.
    """
    train_df = pd.read_parquet(cfg.paths.train)
    val_df = pd.read_parquet(cfg.paths.val)

    dl = cfg.data_load
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

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    return train_df, val_df
