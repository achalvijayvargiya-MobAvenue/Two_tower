from __future__ import annotations

import io
import math
import pickle
import time
import os
import contextlib
from pathlib import Path
from typing import Union

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from two_tower.config_loader import load_pipeline_config
from two_tower.configs import PipelineConfig
from two_tower.data.dataset import TwoTowerDataset
from two_tower.data.balanced_batch_sampler import BalancedClientLabelBatchSampler
from two_tower.data.load import load_train_validation_frames
from two_tower.features.encode import collate_fn
from two_tower.features.prepare import FeatureArtifacts, prepare_training_features
from two_tower.features.vocab import vocab_to_dict
from two_tower.io.paths import artifact_uri
from two_tower.io.runlog import start_run_log
from two_tower.mlflow_utils import setup_mlflow
from two_tower.model.two_tower import build_two_tower_model, embedding_dim_for_cardinality


def _resolve_device(train_device: str) -> torch.device:
    if train_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(train_device)


def _dist_env() -> tuple[bool, int, int, int]:
    """
    Read torchrun/SageMaker DDP env vars.

    Returns: (is_distributed, rank, world_size, local_rank)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SM_LOCAL_RANK", "0")) or "0")
    return (world_size > 1, rank, world_size, local_rank)


def _is_rank0() -> bool:
    is_dist, rank, _, _ = _dist_env()
    return (not is_dist) or (rank == 0)


def _all_gather_1d(x: torch.Tensor) -> torch.Tensor:
    """
    All-gather a 1D tensor across ranks and return concatenated tensor.

    Important: for NCCL backend this MUST run on CUDA tensors (not CPU).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.contiguous()
    ws = dist.get_world_size()
    device = x.device
    n_local = torch.tensor([x.numel()], dtype=torch.long, device=device)
    sizes = [torch.zeros((1,), dtype=torch.long, device=device) for _ in range(ws)]
    dist.all_gather(sizes, n_local)
    sizes_i = [int(s.item()) for s in sizes]
    max_n = max(sizes_i) if sizes_i else int(x.numel())
    padded = torch.zeros((max_n,), dtype=x.dtype, device=device)
    padded[: x.numel()] = x
    gathered = [torch.empty((max_n,), dtype=x.dtype, device=device) for _ in range(ws)]
    dist.all_gather(gathered, padded)
    out = [g[:n] for g, n in zip(gathered, sizes_i)]
    return torch.cat(out, dim=0)


def _write_bytes(uri: str, data: bytes) -> None:
    if uri.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem()
        with fs.open(uri, "wb") as f:
            f.write(data)
    else:
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _resolve_client_id_column(train_df: pd.DataFrame, preferred: str) -> str:
    candidates = [
        preferred,
        "client_id",
        "client_bundle_id",
        "client_bundle",
        "clientId",
        "bundle_id",
        "client",
    ]
    for c in candidates:
        if c in train_df.columns:
            return c
    raise KeyError(
        "Could not find client id column in training data for client embedding export. "
        f"Tried: {candidates}. Available: {list(train_df.columns)[:50]} ..."
    )


def _sigmoid_prob_from_logits(y_score: np.ndarray) -> np.ndarray:
    # Clamp for numerical stability (avoids overflow in exp).
    z = np.clip(np.asarray(y_score, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a compact set of binary classification metrics from labels and scores.

    - y_true: 0/1 labels (any numeric dtype ok)
    - y_score: model scores (logits or probabilities are fine; AUCs treat as ranking)
    """
    yt = np.asarray(y_true).astype(np.int32).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()

    # For probability-based metrics, interpret y_score as logits (current training loop output).
    prob = _sigmoid_prob_from_logits(ys)
    pred = (prob >= float(threshold)).astype(np.int32)

    out: dict[str, float] = {}
    # Ranking metrics
    try:
        out["auc_roc"] = float(roc_auc_score(yt, ys))
    except ValueError:
        out["auc_roc"] = float("nan")
    try:
        out["auc_pr"] = float(average_precision_score(yt, ys))
    except ValueError:
        out["auc_pr"] = float("nan")

    # Threshold metrics
    out["precision"] = float(precision_score(yt, pred, zero_division=0))
    out["recall"] = float(recall_score(yt, pred, zero_division=0))
    out["f1"] = float(f1_score(yt, pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(yt, pred))

    # Calibration-ish / probabilistic
    try:
        out["logloss"] = float(log_loss(yt, prob, labels=[0, 1]))
    except ValueError:
        out["logloss"] = float("nan")

    # Class balance + confusion matrix (useful for "is it predicting all zeros?")
    out["label_pos_rate"] = float(yt.mean()) if yt.size else float("nan")
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    out["tn"] = float(cm[0, 0])
    out["fp"] = float(cm[0, 1])
    out["fn"] = float(cm[1, 0])
    out["tp"] = float(cm[1, 1])
    return out


def _safe_mlflow_key(s: object) -> str:
    """
    Make a stable MLflow metric key suffix from a client id (may contain slashes/dots/spaces).
    """
    raw = str(s)
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned[:120] if cleaned else "unknown"


def _client_group_metrics(
    *,
    val_df: pd.DataFrame,
    client_id_col: str,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[dict[object, dict[str, float]], dict[str, float]]:
    """
    Compute per-client metrics by mapping each scored example back to its client id via val_df[row_idx].
    Returns:
      - per_client: {client_id: metrics_dict}
      - summary: {"macro_auc": ..., "worst_client_auc": ...}
    """
    if client_id_col not in val_df.columns:
        return {}, {"macro_auc": float("nan"), "worst_client_auc": float("nan")}

    idx = np.asarray(row_idx, dtype=np.int64).ravel()
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score).ravel()
    if idx.size != yt.size or idx.size != ys.size:
        raise ValueError(
            f"client metrics: size mismatch row_idx={idx.size} y_true={yt.size} y_score={ys.size}"
        )

    # val_df is reset_index(drop=True) inside the dataset; row_idx refers to that 0..N-1 space.
    client_ids = val_df[client_id_col].reset_index(drop=True).to_numpy()
    cid_for_row = client_ids[idx]

    per_client: dict[object, dict[str, float]] = {}
    aucs: list[float] = []
    for cid in pd.unique(cid_for_row):
        m = (cid_for_row == cid)
        if not bool(np.any(m)):
            continue
        met = _binary_metrics(yt[m], ys[m], threshold=0.5)
        met["_n"] = float(int(np.sum(m)))
        per_client[cid] = met
        a = float(met.get("auc_roc", float("nan")))
        if np.isfinite(a):
            aucs.append(a)

    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    worst_auc = float(np.min(aucs)) if aucs else float("nan")
    return per_client, {"macro_auc": macro_auc, "worst_client_auc": worst_auc}


def train_and_log(
    *,
    cfg: Union[PipelineConfig, str, Path],
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    feature_artifacts: FeatureArtifacts | None = None,
) -> None:
    runlog = start_run_log(kind="train", name="two_tower")
    t_start = time.time()
    if not isinstance(cfg, PipelineConfig):
        cfg = load_pipeline_config(cfg)

    is_dist, rank, world_size, local_rank = _dist_env()

    if train_df is None or val_df is None:
        train_df, val_df = load_train_validation_frames(cfg)

    if _is_rank0():
        print(f"[train_and_log] train={train_df.shape} val={val_df.shape}")
        print(f"[train_and_log] run log: {runlog.path}")
        runlog.write(
            f"CONFIG experiment={cfg.train.experiment_name} device={cfg.train.device} "
            f"epochs={cfg.train.epochs} world_size={world_size}"
        )
        runlog.write(f"DATA train_rows={train_df.shape[0]} val_rows={val_df.shape[0]}")

    if feature_artifacts is None:
        feature_artifacts = prepare_training_features(train_df, val_df, cfg)

    fa = feature_artifacts
    tc = cfg.train
    fc = cfg.features

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed % (2**32 - 1))

    device = _resolve_device(tc.device)
    if is_dist:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend=backend)
    # Ensure stable 0..N-1 indexing because the dataset resets index(drop=True).
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = TwoTowerDataset(train_df, fa, label_col=fc.label_col, multi_max_tokens=tc.multi_max_tokens)
    val_ds = TwoTowerDataset(val_df, fa, label_col=fc.label_col, multi_max_tokens=tc.multi_max_tokens)

    # Keep global batch size constant across DDP ranks.
    if is_dist:
        if tc.batch_size % world_size != 0:
            raise ValueError(
                f"train.batch_size (global) must be divisible by world_size. "
                f"Got batch_size={tc.batch_size} world_size={world_size}."
            )
        per_rank_bs = tc.batch_size // world_size
    else:
        per_rank_bs = tc.batch_size

    pin_memory = bool(getattr(tc, "dataloader_pin_memory", True)) and (device.type == "cuda")
    persistent_workers = bool(getattr(tc, "dataloader_persistent_workers", True)) and (tc.num_workers > 0)
    prefetch_factor = int(getattr(tc, "dataloader_prefetch_factor", 2))
    prefetch_factor = max(prefetch_factor, 1)

    train_sampler = None
    val_sampler = None
    if is_dist:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=tc.seed)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=tc.seed)

    batch_sampler = None
    if bool(getattr(tc, "batch_balance", False)):
        if is_dist:
            # Implementing globally-correct per-batch ratios under DDP requires a coordinated sampler.
            # For now, fall back to DistributedSampler.
            if _is_rank0():
                print("[train] batch_balance requested but DDP is enabled; falling back to standard sampling.")
        else:
            batch_sampler = BalancedClientLabelBatchSampler(
                train_df,
                client_id_col=fc.client_id_col,
                label_col=fc.label_col,
                batch_size=per_rank_bs,
                neg_per_pos=int(getattr(tc, "batch_balance_neg_per_pos", 3)),
                seed=int(tc.seed),
            )

    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=tc.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if tc.num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=per_rank_bs,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=tc.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if tc.num_workers > 0 else None,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_rank_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=tc.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if tc.num_workers > 0 else None,
    )

    user_vocab_sizes = [fa.user_vocabs[c].size for c in fa.user_cat_cols]
    client_vocab_sizes = [fa.client_vocabs[c].size for c in fa.client_cat_cols]
    user_multi_vocab_sizes = [fa.user_multi_vocabs[c].size for c in fa.user_multi_cols]
    client_multi_vocab_sizes = [fa.client_multi_vocabs[c].size for c in fa.client_multi_cols]
    user_multi_emb_dims = [embedding_dim_for_cardinality(v) for v in user_multi_vocab_sizes]
    client_multi_emb_dims = [embedding_dim_for_cardinality(v) for v in client_multi_vocab_sizes]
    user_num_dim = len(fa.user_num_cols)
    client_num_dim = len(fa.client_num_cols)
    user_cat_out = sum(embedding_dim_for_cardinality(v) for v in user_vocab_sizes)
    client_cat_out = sum(embedding_dim_for_cardinality(v) for v in client_vocab_sizes)
    user_input_dim = user_cat_out + sum(user_multi_emb_dims) + user_num_dim
    client_input_dim = client_cat_out + sum(client_multi_emb_dims) + client_num_dim

    use_pt = bool(fa.user_cat_pretrained_weights) and bool(fa.client_cat_pretrained_weights)
    model = build_two_tower_model(fa, cfg).to(device)

    if getattr(tc, "torch_compile", False):
        try:
            model = torch.compile(model, mode=str(getattr(tc, "torch_compile_mode", "reduce-overhead")))
        except Exception as e:
            if _is_rank0():
                print(f"torch.compile disabled due to error: {e!r}")

    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    m = model.module if hasattr(model, "module") else model

    if use_pt and _is_rank0():
        n_frozen = sum(not any(p.requires_grad for p in emb.parameters()) for emb in m.user_tower.user_cat.embs)
        print(
            f"Model: pretrained cat embeddings ({'frozen' if tc.freeze_pretrained_base else 'fine-tunable'} base, "
            f"{tc.pretrained_cat_emb_dim}-dim projection). Frozen user emb tables: {n_frozen}/{len(m.user_tower.user_cat.embs)}."
        )
    elif _is_rank0():
        print("Model: random-init categorical embeddings.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    if _is_rank0():
        setup_mlflow(cfg.mlflow_tracking_uri, tc.experiment_name)

    try:
        mlflow_ctx = mlflow.start_run(run_name=tc.run_name) if _is_rank0() else contextlib.nullcontext()
        with mlflow_ctx:
            if _is_rank0():
                mlflow.log_params(
                    {
                        "batch_size_global": tc.batch_size,
                        "batch_size_per_rank": per_rank_bs,
                        "world_size": world_size,
                    "lr": tc.lr,
                    "weight_decay": tc.weight_decay,
                    "epochs": tc.epochs,
                    "emb_dim": tc.embed_dim,
                    "num_cross_layers": tc.dcn_cross_layers,
                    "user_deep_hidden": str(list(tc.mlp_hidden_dims)),
                    "client_mlp_hidden": str(list(tc.client_mlp_hidden)),
                    "log_scale_init": round(math.log(20.0), 4),
                    "user_input_dim": user_input_dim,
                    "client_input_dim": client_input_dim,
                    "min_token_count": tc.min_count,
                    "num_oov_buckets": tc.num_oov_buckets,
                    "multi_cat_max_tokens": tc.multi_max_tokens,
                    "multi_cat_pool": tc.multi_cat_pool,
                    "pretrained_emb_dim": tc.pretrained_emb_dim,
                    "pretrained_cat_emb_dim": tc.pretrained_cat_emb_dim,
                    "freeze_pretrained_base": tc.freeze_pretrained_base,
                    "use_pretrained_cat": use_pt,
                    "torch_compile": bool(getattr(tc, "torch_compile", False)),
                    "torch_compile_mode": str(getattr(tc, "torch_compile_mode", "reduce-overhead")),
                    }
                )

            best_val_auc = float("-inf")
            best_epoch = -1
            best_snapshot: dict | None = None

            last_hb = time.time()
            hb_every_s = 300.0  # 5 minutes
            for epoch in range(tc.epochs):
                model.train()
                total_loss = 0.0
                total_count = 0
                # Optional: compute train AUC/precision/recall on a bounded sample of seen examples.
                # This helps diagnose overfitting (train >> val) without storing the full epoch.
                train_eval_max = int(getattr(tc, "train_eval_max_examples", 0) or 0)
                train_eval_max = max(train_eval_max, 0)
                train_scores: list[np.ndarray] = []
                train_labels: list[np.ndarray] = []
                train_row_idx: list[np.ndarray] = []
                train_eval_kept = 0

                if is_dist and train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                steps_per_epoch = getattr(tc, "steps_per_epoch", None)
                if steps_per_epoch is None:
                    batch_iter = iter(train_loader)
                    steps_per_epoch = None
                else:
                    steps_per_epoch = int(steps_per_epoch)
                    if steps_per_epoch <= 0:
                        raise ValueError("train.steps_per_epoch must be > 0 when set")
                    batch_iter = iter(train_loader)

                step = 0
                while True:
                    if steps_per_epoch is not None and step >= steps_per_epoch:
                        break
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(train_loader)
                        batch = next(batch_iter)

                    nb = bool(pin_memory)
                    user_cat = batch.user_cat.to(device, non_blocking=nb)
                    user_num = batch.user_num.to(device, non_blocking=nb)
                    user_multi = batch.user_multi.to(device, non_blocking=nb)
                    client_cat = batch.client_cat.to(device, non_blocking=nb)
                    client_num = batch.client_num.to(device, non_blocking=nb)
                    client_multi = batch.client_multi.to(device, non_blocking=nb)
                    labels = batch.label.to(device, non_blocking=nb)

                    optimizer.zero_grad(set_to_none=True)
                    logits, _, _ = model(user_cat, user_num, client_cat, client_num, user_multi, client_multi)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    bs = labels.size(0)
                    total_loss += loss.item() * bs
                    total_count += bs

                    if train_eval_max > 0 and train_eval_kept < train_eval_max:
                        # Keep up to train_eval_max examples (first-N) for stable, cheap epoch metrics.
                        keep = min(bs, train_eval_max - train_eval_kept)
                        if keep > 0:
                            train_scores.append(logits.detach().cpu().numpy().ravel()[:keep])
                            train_labels.append(labels.detach().cpu().numpy().ravel()[:keep])
                            train_row_idx.append(batch.row_idx.detach().cpu().numpy().ravel()[:keep])
                            train_eval_kept += keep
                    now = time.time()
                    if now - last_hb >= hb_every_s:
                        if _is_rank0():
                            runlog.write(
                                f"HEARTBEAT epoch={epoch + 1}/{tc.epochs} seen={total_count} "
                                f"avg_loss={total_loss / max(total_count, 1):.6f}"
                            )
                        last_hb = now
                    step += 1

                avg_train_loss = total_loss / max(total_count, 1)
                train_metrics: dict[str, float] | None = None
                train_client_metrics: dict[object, dict[str, float]] = {}
                train_client_summary: dict[str, float] = {"macro_auc": float("nan"), "worst_client_auc": float("nan")}
                if train_eval_max > 0 and train_eval_kept > 0 and _is_rank0():
                    ytr = np.concatenate(train_labels, axis=0)
                    yts = np.concatenate(train_scores, axis=0)
                    train_metrics = _binary_metrics(ytr, yts, threshold=0.5)
                    # Optional per-client train metrics on the same sample (maps back via row_idx).
                    ridx = np.concatenate(train_row_idx, axis=0)
                    max_c = getattr(tc, "train_client_eval_max_examples", None)
                    if max_c is None:
                        max_c = train_eval_max
                    max_c = int(max_c or 0)
                    if max_c > 0 and ridx.size > max_c:
                        ridx = ridx[:max_c]
                        ytr = ytr[:max_c]
                        yts = yts[:max_c]
                    train_client_metrics, train_client_summary = _client_group_metrics(
                        val_df=train_df,
                        client_id_col=fc.client_id_col,
                        row_idx=ridx,
                        y_true=ytr,
                        y_score=yts,
                    )

                model.eval()
                val_loss = 0.0
                val_count = 0
                val_uemb_bad = 0
                val_cemb_bad = 0
                val_logits_nonfinite = 0
                all_logits: list[torch.Tensor] = []
                all_labels: list[torch.Tensor] = []
                all_row_idx: list[torch.Tensor] = []
                with torch.no_grad():
                    for batch in val_loader:
                        nb = bool(pin_memory)
                        user_cat = batch.user_cat.to(device, non_blocking=nb)
                        user_num = batch.user_num.to(device, non_blocking=nb)
                        user_multi = batch.user_multi.to(device, non_blocking=nb)
                        client_cat = batch.client_cat.to(device, non_blocking=nb)
                        client_num = batch.client_num.to(device, non_blocking=nb)
                        client_multi = batch.client_multi.to(device, non_blocking=nb)
                        labels = batch.label.to(device, non_blocking=nb)

                        logits, uemb, cemb = model(
                            user_cat, user_num, client_cat, client_num, user_multi, client_multi
                        )
                        loss = criterion(logits, labels)

                        bs = labels.size(0)
                        val_uemb_bad += int((~torch.isfinite(uemb)).any(dim=1).sum().item())
                        val_cemb_bad += int((~torch.isfinite(cemb)).any(dim=1).sum().item())
                        val_logits_nonfinite += int((~torch.isfinite(logits)).sum().item())
                        val_loss += loss.item() * bs
                        val_count += bs
                        all_logits.append(logits.detach().cpu())
                        all_labels.append(labels.detach().cpu())
                        all_row_idx.append(batch.row_idx.detach().cpu())

                avg_val_loss = val_loss / max(val_count, 1)
                local_y_true = torch.cat(all_labels).view(-1)
                local_y_score = torch.cat(all_logits).view(-1)
                local_row_idx = torch.cat(all_row_idx).view(-1)
                if is_dist and dist.is_initialized():
                    # NCCL does not support CPU collectives: gather on the training device.
                    y_true_t = _all_gather_1d(local_y_true.to(device, non_blocking=True)).detach().cpu()
                    y_score_t = _all_gather_1d(local_y_score.to(device, non_blocking=True)).detach().cpu()
                    ridx_t = _all_gather_1d(local_row_idx.to(device, non_blocking=True)).detach().cpu()
                else:
                    y_true_t = local_y_true
                    y_score_t = local_y_score
                    ridx_t = local_row_idx

                if _is_rank0():
                    y_true = y_true_t.numpy()
                    y_score = y_score_t.numpy()
                    row_idx = ridx_t.numpy()
                    val_metrics = _binary_metrics(y_true, y_score, threshold=0.5)
                    val_auc = float(val_metrics["auc_roc"])
                    per_client_metrics, client_summary = _client_group_metrics(
                        val_df=val_df,
                        client_id_col=fc.client_id_col,
                        row_idx=row_idx,
                        y_true=y_true,
                        y_score=y_score,
                    )
                else:
                    val_metrics = {
                        "auc_roc": float("nan"),
                        "auc_pr": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "f1": float("nan"),
                        "accuracy": float("nan"),
                        "logloss": float("nan"),
                        "label_pos_rate": float("nan"),
                        "tn": float("nan"),
                        "fp": float("nan"),
                        "fn": float("nan"),
                        "tp": float("nan"),
                    }
                    val_auc = float("nan")
                    per_client_metrics = {}
                    client_summary = {"macro_auc": float("nan"), "worst_client_auc": float("nan")}

                if np.isfinite(val_auc) and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    best_snapshot = {
                        "user_tower": {
                            k: v.detach().cpu().clone() for k, v in m.user_tower.state_dict().items()
                        },
                        "client_tower": {
                            k: v.detach().cpu().clone() for k, v in m.client_tower.state_dict().items()
                        },
                        "log_scale": m.log_scale.detach().cpu().clone(),
                    }
                    if _is_rank0():
                        print(f"  -> new best val_auc={best_val_auc:.4f} (epoch {epoch + 1})")

            _scale_val = m.log_scale.clamp(math.log(1.0), math.log(100.0)).exp().item()
            metrics_to_log = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_uemb_nonfinite_rows": float(val_uemb_bad),
                "val_cemb_nonfinite_rows": float(val_cemb_bad),
                "val_logits_nonfinite_cells": float(val_logits_nonfinite),
                "temperature_scale": _scale_val,
                # validation (rich)
                "val_auc": float(val_metrics["auc_roc"]),
                "val_auc_pr": float(val_metrics["auc_pr"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_logloss": float(val_metrics["logloss"]),
                "val_label_pos_rate": float(val_metrics["label_pos_rate"]),
                "val_tn": float(val_metrics["tn"]),
                "val_fp": float(val_metrics["fp"]),
                "val_fn": float(val_metrics["fn"]),
                "val_tp": float(val_metrics["tp"]),
                "val_auc_macro_client": float(client_summary["macro_auc"]),
                "val_auc_worst_client": float(client_summary["worst_client_auc"]),
            }
            if _is_rank0() and per_client_metrics:
                # Log per-client metrics for debugging/fairness tracking.
                for cid, met in per_client_metrics.items():
                    k = _safe_mlflow_key(cid)
                    metrics_to_log[f"val_client__{k}__auc"] = float(met.get("auc_roc", float("nan")))
                    metrics_to_log[f"val_client__{k}__auc_pr"] = float(met.get("auc_pr", float("nan")))
                    metrics_to_log[f"val_client__{k}__precision"] = float(met.get("precision", float("nan")))
                    metrics_to_log[f"val_client__{k}__recall"] = float(met.get("recall", float("nan")))
                    metrics_to_log[f"val_client__{k}__f1"] = float(met.get("f1", float("nan")))
                    metrics_to_log[f"val_client__{k}__logloss"] = float(met.get("logloss", float("nan")))
                    metrics_to_log[f"val_client__{k}__pos_rate"] = float(met.get("label_pos_rate", float("nan")))
                    metrics_to_log[f"val_client__{k}__n"] = float(met.get("_n", float("nan")))
            if train_metrics is not None:
                metrics_to_log.update(
                    {
                        "train_auc": float(train_metrics["auc_roc"]),
                        "train_auc_pr": float(train_metrics["auc_pr"]),
                        "train_precision": float(train_metrics["precision"]),
                        "train_recall": float(train_metrics["recall"]),
                        "train_f1": float(train_metrics["f1"]),
                        "train_accuracy": float(train_metrics["accuracy"]),
                        "train_logloss": float(train_metrics["logloss"]),
                        "train_label_pos_rate": float(train_metrics["label_pos_rate"]),
                        "train_auc_macro_client": float(train_client_summary["macro_auc"]),
                        "train_auc_worst_client": float(train_client_summary["worst_client_auc"]),
                    }
                )
                if _is_rank0() and train_client_metrics:
                    for cid, met in train_client_metrics.items():
                        k = _safe_mlflow_key(cid)
                        metrics_to_log[f"train_client__{k}__auc"] = float(met.get("auc_roc", float("nan")))
                        metrics_to_log[f"train_client__{k}__auc_pr"] = float(met.get("auc_pr", float("nan")))
                        metrics_to_log[f"train_client__{k}__precision"] = float(met.get("precision", float("nan")))
                        metrics_to_log[f"train_client__{k}__recall"] = float(met.get("recall", float("nan")))
                        metrics_to_log[f"train_client__{k}__f1"] = float(met.get("f1", float("nan")))
                        metrics_to_log[f"train_client__{k}__logloss"] = float(met.get("logloss", float("nan")))
                        metrics_to_log[f"train_client__{k}__pos_rate"] = float(met.get("label_pos_rate", float("nan")))
                        metrics_to_log[f"train_client__{k}__n"] = float(met.get("_n", float("nan")))
            if _is_rank0():
                mlflow.log_metrics(metrics_to_log, step=epoch)

            if _is_rank0():
                print(
                    f"Epoch {epoch + 1}/{tc.epochs} | train_loss={avg_train_loss:.4f} | "
                    f"val_loss={avg_val_loss:.4f} | val_auc={val_metrics['auc_roc']:.4f} | "
                    f"val_auc_macro={client_summary['macro_auc']:.4f} "
                    f"val_auc_worst={client_summary['worst_client_auc']:.4f} | "
                    f"val_pr_auc={val_metrics['auc_pr']:.4f} | "
                    f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f} | scale={_scale_val:.2f} | "
                    f"val_nonfinite uemb={val_uemb_bad} cemb={val_cemb_bad} logits={val_logits_nonfinite}"
                )
                if train_metrics is not None:
                    print(
                        f"           train_auc={train_metrics['auc_roc']:.4f} train_pr_auc={train_metrics['auc_pr']:.4f} "
                        f"train_prec={train_metrics['precision']:.4f} train_rec={train_metrics['recall']:.4f} "
                        f"train_f1={train_metrics['f1']:.4f} "
                        f"train_auc_macro={train_client_summary['macro_auc']:.4f} "
                        f"train_auc_worst={train_client_summary['worst_client_auc']:.4f} "
                        f"(sample_n={train_eval_kept})"
                    )
                runlog.write(
                    "EPOCH_DONE "
                    f"epoch={epoch + 1}/{tc.epochs} "
                    f"train_loss={avg_train_loss:.6f} "
                    f"val_loss={avg_val_loss:.6f} "
                    f"val_auc={val_metrics['auc_roc']:.6f} val_auc_pr={val_metrics['auc_pr']:.6f} "
                    f"val_prec={val_metrics['precision']:.6f} val_rec={val_metrics['recall']:.6f} "
                    f"val_f1={val_metrics['f1']:.6f} val_acc={val_metrics['accuracy']:.6f} "
                    f"val_logloss={val_metrics['logloss']:.6f} "
                    f"val_cm_tn={int(val_metrics['tn'])} val_cm_fp={int(val_metrics['fp'])} "
                    f"val_cm_fn={int(val_metrics['fn'])} val_cm_tp={int(val_metrics['tp'])}"
                )
                if per_client_metrics:
                    parts = []
                    for cid, met in per_client_metrics.items():
                        parts.append(
                            f"{_safe_mlflow_key(cid)}:"
                            f"n={int(float(met.get('_n', float('nan')))) if np.isfinite(float(met.get('_n', float('nan')))) else -1}"
                            f" pos={float(met.get('label_pos_rate', float('nan'))):.4f}"
                            f" auc={float(met.get('auc_roc', float('nan'))):.4f}"
                        )
                    runlog.write("VAL_CLIENT_AUC " + " ".join(parts))
                if train_client_metrics:
                    parts = []
                    for cid, met in train_client_metrics.items():
                        parts.append(
                            f"{_safe_mlflow_key(cid)}:"
                            f"n={int(float(met.get('_n', float('nan')))) if np.isfinite(float(met.get('_n', float('nan')))) else -1}"
                            f" pos={float(met.get('label_pos_rate', float('nan'))):.4f}"
                            f" auc={float(met.get('auc_roc', float('nan'))):.4f}"
                        )
                    runlog.write("TRAIN_CLIENT_AUC " + " ".join(parts))

        if _is_rank0():
            if best_snapshot is not None:
                best_user_state = best_snapshot["user_tower"]
                best_client_state = best_snapshot["client_tower"]
                ckpt_selection = "best_val_auc"
                mlflow.log_metrics({"best_val_auc": best_val_auc, "best_epoch": float(best_epoch)}, step=tc.epochs)
                print(
                    f"Best epoch: {best_epoch + 1}/{tc.epochs} val_auc={best_val_auc:.4f} "
                    "(weights exported to artifacts + MLflow)."
                )
            else:
                best_user_state = {k: v.detach().cpu().clone() for k, v in m.user_tower.state_dict().items()}
                best_client_state = {k: v.detach().cpu().clone() for k, v in m.client_tower.state_dict().items()}
                ckpt_selection = "last_epoch"
                print("No val_auc improvement recorded; exporting last-epoch towers.")

            m.user_tower.load_state_dict({k: v.to(device) for k, v in best_user_state.items()})
            m.client_tower.load_state_dict({k: v.to(device) for k, v in best_client_state.items()})
            if best_snapshot is not None and "log_scale" in best_snapshot:
                m.log_scale.data.copy_(best_snapshot["log_scale"].to(device))

            _ex = next(iter(val_loader))
            input_example = [
                _ex.user_cat.numpy(),
                _ex.user_num.numpy(),
                _ex.user_multi.numpy(),
                _ex.client_cat.numpy(),
                _ex.client_num.numpy(),
                _ex.client_multi.numpy(),
            ]
            try:
                mlflow.pytorch.log_model(
                    m,
                    name="model",
                    input_example=input_example,
                    serialization_format="pt2",
                )
            except Exception as e:
                print(f"MLflow pt2 export failed ({e!r}); falling back to pickle serialization.")
                mlflow.pytorch.log_model(m, name="model", input_example=input_example)

            base = cfg.paths.artifacts_base
            user_tower_uri = artifact_uri(base, "artifacts", "user_tower", "user_tower_state.pt")
            vocab_uri = artifact_uri(base, "artifacts", "vocab_artifact", "vocab_artifact.pkl")
            client_emb_uri = artifact_uri(base, "artifacts", "client_embeddings", "client_embeddings.parquet")

            user_multi_vocab_sizes = [fa.user_multi_vocabs[c].size for c in fa.user_multi_cols]
            user_multi_emb_dims = [embedding_dim_for_cardinality(v) for v in user_multi_vocab_sizes]

            user_ckpt = {
                "state_dict": best_user_state,
                "emb_dim": tc.embed_dim,
                "user_vocab_sizes": user_vocab_sizes,
                "user_num_dim": user_num_dim,
                "user_multi_vocab_sizes": user_multi_vocab_sizes,
                "user_multi_emb_dims": user_multi_emb_dims,
                "user_multi_cols": fa.user_multi_cols,
                "multi_cat_max_tokens": tc.multi_max_tokens,
                "multi_cat_pool": tc.multi_cat_pool,
                "num_cross_layers": tc.dcn_cross_layers,
                "user_deep_hidden": list(tc.mlp_hidden_dims),
                "log_scale": float(m.log_scale.item()),
                "checkpoint_selection": ckpt_selection,
                "best_val_auc": float(best_val_auc) if best_snapshot is not None else None,
                "best_epoch_0based": int(best_epoch) if best_snapshot is not None else None,
                "use_pretrained_cat": use_pt,
                "pretrained_emb_dim": tc.pretrained_emb_dim,
                "target_cat_emb_dim": tc.pretrained_cat_emb_dim,
                "freeze_base": tc.freeze_pretrained_base,
            }
            _ubuf = io.BytesIO()
            torch.save(user_ckpt, _ubuf)
            _write_bytes(user_tower_uri, _ubuf.getvalue())
            print(f"Saved user tower to: {user_tower_uri} (selection={ckpt_selection})")

            vocab_artifact = {
                "user_vocabs": {k: vocab_to_dict(v) for k, v in fa.user_vocabs.items()},
                "user_multi_vocabs": {k: vocab_to_dict(v) for k, v in fa.user_multi_vocabs.items()},
                "user_cat_cols": fa.user_cat_cols,
                "user_num_cols": fa.user_num_cols,
                "user_multi_cols": fa.user_multi_cols,
                "device_id_col": fc.device_id_col,
                "multi_cat_max_tokens": tc.multi_max_tokens,
                "multi_cat_pool": tc.multi_cat_pool,
            }
            _vbuf = io.BytesIO()
            pickle.dump(vocab_artifact, _vbuf, protocol=pickle.HIGHEST_PROTOCOL)
            _write_bytes(vocab_uri, _vbuf.getvalue())
            print(f"Saved vocab artifact to: {vocab_uri}")

            client_id_col = _resolve_client_id_column(train_df, fc.client_id_col)
            client_df = train_df.drop_duplicates(subset=[client_id_col]).copy()
            client_df["client_id"] = client_df[client_id_col]

            from two_tower.features.encode import encode_cats, encode_multi_matrix, encode_nums

            m.client_tower.eval()
            _client_cat = encode_cats(client_df, fa.client_cat_cols, fa.client_vocabs)
            _client_num = encode_nums(client_df, fa.client_num_cols)
            _client_multi = encode_multi_matrix(
                client_df, fa.client_multi_cols, fa.client_multi_vocabs, tc.multi_max_tokens
            )
            embs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(client_df), tc.batch_size):
                    end = start + tc.batch_size
                    cc = _client_cat[start:end].to(device)
                    cn = _client_num[start:end].to(device)
                    cm = _client_multi[start:end].to(device)
                    emb = m.client_tower(cc, cn, cm).cpu().numpy().astype("float32")
                    embs.append(emb)
            client_emb_matrix = np.concatenate(embs, axis=0)
            out_df = pd.DataFrame({"client_id": client_df["client_id"].values})
            out_df["embedding"] = [row.tolist() for row in client_emb_matrix]
            out_df.to_parquet(client_emb_uri, index=False)
            print(f"Wrote client embeddings to: {client_emb_uri}")

            print("[train_and_log] Done.")
            runlog.write(f"FINISH ok=true elapsed_s={time.time() - t_start:.1f}")
    except Exception as e:
        if _is_rank0():
            runlog.write(f"FINISH ok=false elapsed_s={time.time() - t_start:.1f} error={e!r}")
        raise
    finally:
        if is_dist and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
