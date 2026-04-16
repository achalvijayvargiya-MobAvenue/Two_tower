from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataPaths:
    train: str
    val: str
    infer: str
    artifacts_base: str  # e.g. s3://.../two_tower_ml or local path


@dataclass(frozen=True)
class FeatureConfig:
    label_col: str
    device_id_col: str
    client_id_col: str
    user_feature_cols: list[str]
    client_feature_cols: list[str]
    user_multi_cols: list[str]
    client_multi_cols: list[str]


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    run_name: str | None
    seed: int

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float

    embed_dim: int  # final embedding dim (e.g. 1024)
    dcn_cross_layers: int
    mlp_hidden_dims: list[int]

    min_count: int
    num_oov_buckets: int
    multi_max_tokens: int

    num_workers: int
    device: str  # "cuda" or "cpu"

    # Hash embedding init for categorical columns (reference: PRETRAINED_EMB_DIM)
    pretrained_emb_dim: int = 128
    pretrained_cat_emb_dim: int = 64
    freeze_pretrained_base: bool = True

    # Multi-value categorical pooling in towers (reference: MULTI_CAT_POOL)
    multi_cat_pool: str = "mean"

    # Client MLP hidden layers (reference: CLIENT_HIDDEN); user deep stack uses mlp_hidden_dims
    client_mlp_hidden: list[int] = field(default_factory=lambda: [256, 256])


@dataclass(frozen=True)
class InferenceConfig:
    topk_clients: int
    infer_stream_batch_rows: int

    # worker pool / GPU sharding
    num_physical_gpus: int
    workers_per_gpu: int

    # outputs
    ranking_output: str

    # batching / perf (reference: RANK_USER_BATCH, CLIENT_CHUNK, etc.)
    rank_user_batch: int = 4096
    client_chunk: int = 8192
    use_amp: bool = True
    amp_dtype: str = "float16"
    output_min_rows_per_part: int = 100_000
    output_parquet_compression: str = "zstd"
    debug_cuda: bool = False

    # Optional run-size controls for quick smoke tests.
    # - max_files: only process first N parquet inputs under paths.infer
    # - max_users_per_file: stop after ranking N unique users per parquet file
    max_files: int | None = None
    max_users_per_file: int | None = None


@dataclass(frozen=True)
class InferPaths:
    """Minimal paths for a standalone inference job (`configs/infer.yaml`)."""

    infer: str
    artifacts_base: str


@dataclass(frozen=True)
class InferJobConfig:
    paths: InferPaths
    infer: InferenceConfig


@dataclass(frozen=True)
class DataLoadConfig:
    """Optional preprocessing when loading train/val (matches reference notebook)."""

    inject_single_client_metadata: bool = False
    single_client_metadata_uri: str | None = None
    single_client_row_filter: dict[str, Any] | None = None
    single_client_features_hardcoded: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineConfig:
    paths: DataPaths
    features: FeatureConfig
    train: TrainConfig
    infer: InferenceConfig
    data_load: DataLoadConfig = field(default_factory=DataLoadConfig)

    mlflow_tracking_uri: str | None = None
    extra: dict[str, Any] | None = None

