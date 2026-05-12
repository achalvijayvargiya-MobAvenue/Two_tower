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
    user_mlp_hidden: list[int]

    min_count: int
    num_oov_buckets: int
    multi_max_tokens: int

    num_workers: int
    device: str  # "cuda" or "cpu"

    # If set, run a fixed number of optimizer steps per epoch (option B training).
    # When unset, an epoch iterates over the DataLoader once.
    steps_per_epoch: int | None = None

    # Optional: compute train metrics each epoch on a bounded sample of seen examples.
    # Set to 0/None to disable.
    train_eval_max_examples: int | None = None
    # Optional: per-client train metrics sample size (defaults to train_eval_max_examples if unset).
    train_client_eval_max_examples: int | None = None

    # Performance / distributed (all optional; defaults preserve current behavior)
    # If torch.distributed env vars are set (WORLD_SIZE>1), training will use DDP automatically.
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2

    # Optional compile (PyTorch 2.x). Keeps FP32 math; can be disabled if unstable.
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    # Optional per-client downsampling (structured version of notebook Step 3a).
    # Applied inside ``load_train_validation_frames`` so it works the same for notebook and DDP.
    downsample_train: bool = False
    downsample_val: bool = False
    # Target negatives per positive within each client (None disables ratio balancing).
    downsample_neg_per_pos: int | None = 10
    # After ratio balancing, cap every client to the same row count (min across clients).
    downsample_equalize_client_rows: bool = True
    downsample_random_state: int = 42

    # Stop after N consecutive epochs without improvement in val ROC-AUC (None / 0 = disabled).
    early_stopping_patience: int | None = None

    # Batch-level balancing (does NOT change dataset size; affects only sampling into batches).
    # When enabled, batches are constructed to match global per-client frequency and enforce a
    # per-client negative-to-positive ratio (e.g. 3 means 3 negatives per positive).
    batch_balance: bool = False
    batch_balance_neg_per_pos: int = 3

    # Hash embedding init for categorical columns (reference: PRETRAINED_EMB_DIM)
    pretrained_emb_dim: int = 128
    pretrained_cat_emb_dim: int = 64
    freeze_pretrained_base: bool = True

    # Multi-value categorical pooling in towers (reference: MULTI_CAT_POOL)
    multi_cat_pool: str = "mean"

    # Client MLP hidden layers
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
    # User tower runtime backend:
    # - "pytorch": existing nn.Module execution (default)
    # - "tensorrt_onnxruntime": ONNX Runtime with TensorRT EP (falls back to CUDA/CPU EP)
    user_tower_backend: str = "pytorch"
    # Required when user_tower_backend="tensorrt_onnxruntime".
    # Accepts local path or s3:// URI to an ONNX model that takes:
    #   user_cat(int64), user_num(float32), user_multi(int64) -> user embedding(float32)
    user_tower_onnx_uri: str | None = None
    trt_fp16_enable: bool = True
    trt_engine_cache_enable: bool = True
    trt_engine_cache_path: str | None = None

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

    # Per-row join: ``client_metadata`` Parquet (same schema as pipeline ``client_metadata`` table)
    merge_client_metadata: bool = False
    client_metadata_uri: str | None = None

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

