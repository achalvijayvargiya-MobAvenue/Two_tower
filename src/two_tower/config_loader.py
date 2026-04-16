from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from two_tower.configs import (
    DataLoadConfig,
    DataPaths,
    FeatureConfig,
    InferJobConfig,
    InferPaths,
    InferenceConfig,
    PipelineConfig,
    TrainConfig,
)


def _parse_inference_section(i: dict, *, path_label: str) -> InferenceConfig:
    for k in (
        "topk_clients",
        "infer_stream_batch_rows",
        "num_physical_gpus",
        "workers_per_gpu",
        "ranking_output",
    ):
        if k not in i or i[k] is None:
            raise KeyError(f"infer.{k} is required in {path_label}")
    return InferenceConfig(
        topk_clients=int(i["topk_clients"]),
        infer_stream_batch_rows=int(i["infer_stream_batch_rows"]),
        num_physical_gpus=int(i["num_physical_gpus"]),
        workers_per_gpu=int(i["workers_per_gpu"]),
        ranking_output=str(i["ranking_output"]),
        rank_user_batch=int(i.get("rank_user_batch", 4096)),
        client_chunk=int(i.get("client_chunk", 8192)),
        use_amp=bool(i.get("use_amp", True)),
        amp_dtype=str(i.get("amp_dtype", "float16")),
        output_min_rows_per_part=int(i.get("output_min_rows_per_part", 100_000)),
        output_parquet_compression=str(i.get("output_parquet_compression", "zstd")),
        debug_cuda=bool(i.get("debug_cuda", False)),
        max_files=int(i["max_files"]) if i.get("max_files") not in (None, "", 0) else None,
        max_users_per_file=int(i["max_users_per_file"])
        if i.get("max_users_per_file") not in (None, "", 0)
        else None,
    )


def load_infer_job_config(path: str | Path) -> InferJobConfig:
    """Load ``configs/infer.yaml`` (standalone inference)."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)}")
    p = raw.get("paths") or {}
    for k in ("infer", "artifacts_base"):
        if k not in p or p[k] is None:
            raise KeyError(f"paths.{k} is required in {path}")
    paths = InferPaths(infer=str(p["infer"]), artifacts_base=str(p["artifacts_base"]))
    infer = _parse_inference_section(raw.get("infer") or {}, path_label=str(path))
    return InferJobConfig(paths=paths, infer=infer)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load full training pipeline config from YAML (see ``configs/train.yaml``)."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)}")

    p = raw.get("paths") or {}
    for k in ("train", "val", "infer", "artifacts_base"):
        if k not in p or p[k] is None:
            raise KeyError(f"paths.{k} is required in {path}")
    paths = DataPaths(
        train=str(p["train"]),
        val=str(p["val"]),
        infer=str(p["infer"]),
        artifacts_base=str(p["artifacts_base"]),
    )

    f = raw.get("features") or {}
    for k in ("label_col", "device_id_col", "client_id_col"):
        if k not in f or f[k] is None:
            raise KeyError(f"features.{k} is required in {path}")
    features = FeatureConfig(
        label_col=str(f["label_col"]),
        device_id_col=str(f["device_id_col"]),
        client_id_col=str(f["client_id_col"]),
        user_feature_cols=list(f.get("user_feature_cols") or []),
        client_feature_cols=list(f.get("client_feature_cols") or []),
        user_multi_cols=list(f.get("user_multi_cols") or []),
        client_multi_cols=list(f.get("client_multi_cols") or []),
    )

    t = raw.get("train") or {}
    mlp = t.get("mlp_hidden_dims")
    if mlp is None:
        raise KeyError(f"train.mlp_hidden_dims is required in {path}")
    cmh = t.get("client_mlp_hidden")
    if cmh is None:
        cmh = [256, 256]
    train = TrainConfig(
        experiment_name=str(t.get("experiment_name", "two_tower")),
        run_name=t.get("run_name"),
        seed=int(t.get("seed", 42)),
        batch_size=int(t.get("batch_size", 4096)),
        epochs=int(t.get("epochs", 1)),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        embed_dim=int(t.get("embed_dim", 1024)),
        dcn_cross_layers=int(t.get("dcn_cross_layers", 3)),
        mlp_hidden_dims=list(mlp),
        min_count=int(t.get("min_count", 5)),
        num_oov_buckets=int(t.get("num_oov_buckets", 1000)),
        multi_max_tokens=int(t.get("multi_max_tokens", 32)),
        num_workers=int(t.get("num_workers", 4)),
        device=str(t.get("device", "cuda")),
        pretrained_emb_dim=int(t.get("pretrained_emb_dim", 128)),
        pretrained_cat_emb_dim=int(t.get("pretrained_cat_emb_dim", 64)),
        freeze_pretrained_base=bool(t.get("freeze_pretrained_base", True)),
        multi_cat_pool=str(t.get("multi_cat_pool", "mean")),
        client_mlp_hidden=list(cmh),
    )

    infer = _parse_inference_section(raw.get("infer") or {}, path_label=str(path))

    d = raw.get("data") or {}
    row_filter = d.get("single_client_row_filter")
    if row_filter is not None and not isinstance(row_filter, dict):
        raise TypeError("data.single_client_row_filter must be a mapping or null")
    hardcoded = d.get("single_client_features_hardcoded")
    if hardcoded is not None and not isinstance(hardcoded, dict):
        raise TypeError("data.single_client_features_hardcoded must be a mapping or null")

    data_load = DataLoadConfig(
        inject_single_client_metadata=bool(d.get("inject_single_client_metadata", False)),
        single_client_metadata_uri=d.get("single_client_metadata_uri"),
        single_client_row_filter=dict(row_filter) if row_filter else None,
        single_client_features_hardcoded=dict(hardcoded) if hardcoded else None,
    )

    extra = raw.get("extra")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("extra must be a mapping or null")

    return PipelineConfig(
        paths=paths,
        features=features,
        train=train,
        infer=infer,
        data_load=data_load,
        mlflow_tracking_uri=raw.get("mlflow_tracking_uri"),
        extra=dict(extra) if extra else None,
    )
