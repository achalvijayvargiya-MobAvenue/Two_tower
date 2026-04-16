from __future__ import annotations

import io
import math
import pickle
import time
from pathlib import Path
from typing import Union

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from two_tower.config_loader import load_pipeline_config
from two_tower.configs import PipelineConfig
from two_tower.data.dataset import TwoTowerDataset
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

    if train_df is None or val_df is None:
        train_df, val_df = load_train_validation_frames(cfg)

    print(f"[train_and_log] train={train_df.shape} val={val_df.shape}")
    runlog.write(f"CONFIG experiment={cfg.train.experiment_name} device={cfg.train.device} epochs={cfg.train.epochs}")
    runlog.write(f"DATA train_rows={train_df.shape[0]} val_rows={val_df.shape[0]}")

    if feature_artifacts is None:
        feature_artifacts = prepare_training_features(train_df, val_df, cfg)

    fa = feature_artifacts
    tc = cfg.train
    fc = cfg.features

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed % (2**32 - 1))

    device = _resolve_device(tc.device)
    train_ds = TwoTowerDataset(train_df, fa, label_col=fc.label_col, multi_max_tokens=tc.multi_max_tokens)
    val_ds = TwoTowerDataset(val_df, fa, label_col=fc.label_col, multi_max_tokens=tc.multi_max_tokens)

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        collate_fn=collate_fn,
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

    if use_pt:
        n_frozen = sum(not any(p.requires_grad for p in emb.parameters()) for emb in model.user_tower.user_cat.embs)
        print(
            f"Model: pretrained cat embeddings ({'frozen' if tc.freeze_pretrained_base else 'fine-tunable'} base, "
            f"{tc.pretrained_cat_emb_dim}-dim projection). Frozen user emb tables: {n_frozen}/{len(model.user_tower.user_cat.embs)}."
        )
    else:
        print("Model: random-init categorical embeddings.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    setup_mlflow(cfg.mlflow_tracking_uri, tc.experiment_name)

    try:
        with mlflow.start_run(run_name=tc.run_name):
            mlflow.log_params(
                {
                    "batch_size": tc.batch_size,
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

                for batch in train_loader:
                    user_cat = batch.user_cat.to(device)
                    user_num = batch.user_num.to(device)
                    user_multi = batch.user_multi.to(device)
                    client_cat = batch.client_cat.to(device)
                    client_num = batch.client_num.to(device)
                    client_multi = batch.client_multi.to(device)
                    labels = batch.label.to(device)

                    optimizer.zero_grad()
                    logits, _, _ = model(user_cat, user_num, client_cat, client_num, user_multi, client_multi)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    bs = labels.size(0)
                    total_loss += loss.item() * bs
                    total_count += bs
                    now = time.time()
                    if now - last_hb >= hb_every_s:
                        runlog.write(
                            f"HEARTBEAT epoch={epoch + 1}/{tc.epochs} seen={total_count} "
                            f"avg_loss={total_loss / max(total_count, 1):.6f}"
                        )
                        last_hb = now

                avg_train_loss = total_loss / max(total_count, 1)

                model.eval()
                val_loss = 0.0
                val_count = 0
                val_uemb_bad = 0
                val_cemb_bad = 0
                val_logits_nonfinite = 0
                all_logits: list[torch.Tensor] = []
                all_labels: list[torch.Tensor] = []
                with torch.no_grad():
                    for batch in val_loader:
                        user_cat = batch.user_cat.to(device)
                        user_num = batch.user_num.to(device)
                        user_multi = batch.user_multi.to(device)
                        client_cat = batch.client_cat.to(device)
                        client_num = batch.client_num.to(device)
                        client_multi = batch.client_multi.to(device)
                        labels = batch.label.to(device)

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
                        all_logits.append(logits.cpu())
                        all_labels.append(labels.cpu())

                avg_val_loss = val_loss / max(val_count, 1)
                y_true = torch.cat(all_labels).numpy().ravel()
                y_score = torch.cat(all_logits).numpy().ravel()
                try:
                    val_auc = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    val_auc = float("nan")

            _z = np.clip(np.asarray(y_score, dtype=np.float64), -50.0, 50.0)
            val_prob = 1.0 / (1.0 + np.exp(-_z))
            val_pred = (val_prob >= 0.5).astype(np.int32)
            _yt = np.asarray(y_true).astype(np.int32)
            val_precision = float(precision_score(_yt, val_pred, zero_division=0))
            val_recall = float(recall_score(_yt, val_pred, zero_division=0))

            if np.isfinite(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_snapshot = {
                    "user_tower": {k: v.detach().cpu().clone() for k, v in model.user_tower.state_dict().items()},
                    "client_tower": {k: v.detach().cpu().clone() for k, v in model.client_tower.state_dict().items()},
                    "log_scale": model.log_scale.detach().cpu().clone(),
                }
                print(f"  -> new best val_auc={best_val_auc:.4f} (epoch {epoch + 1})")

            _scale_val = model.log_scale.clamp(math.log(1.0), math.log(100.0)).exp().item()
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_auc": val_auc,
                    "val_uemb_nonfinite_rows": val_uemb_bad,
                    "val_cemb_nonfinite_rows": val_cemb_bad,
                    "val_logits_nonfinite_cells": val_logits_nonfinite,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "temperature_scale": _scale_val,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{tc.epochs} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | val_auc={val_auc:.4f} | "
                f"val_prec={val_precision:.4f} val_rec={val_recall:.4f} | scale={_scale_val:.2f} | "
                f"val_nonfinite uemb={val_uemb_bad} cemb={val_cemb_bad} logits={val_logits_nonfinite}"
            )
            runlog.write(
                f"EPOCH_DONE epoch={epoch + 1}/{tc.epochs} train_loss={avg_train_loss:.6f} "
                f"val_loss={avg_val_loss:.6f} val_auc={val_auc:.6f}"
            )

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
            best_user_state = {k: v.detach().cpu().clone() for k, v in model.user_tower.state_dict().items()}
            best_client_state = {k: v.detach().cpu().clone() for k, v in model.client_tower.state_dict().items()}
            ckpt_selection = "last_epoch"
            print("No val_auc improvement recorded; exporting last-epoch towers.")

        model.user_tower.load_state_dict({k: v.to(device) for k, v in best_user_state.items()})
        model.client_tower.load_state_dict({k: v.to(device) for k, v in best_client_state.items()})
        if best_snapshot is not None and "log_scale" in best_snapshot:
            model.log_scale.data.copy_(best_snapshot["log_scale"].to(device))

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
                model,
                name="model",
                input_example=input_example,
                serialization_format="pt2",
            )
        except Exception as e:
            print(f"MLflow pt2 export failed ({e!r}); falling back to pickle serialization.")
            mlflow.pytorch.log_model(model, name="model", input_example=input_example)

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
            "log_scale": float(model.log_scale.item()),
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

        model.client_tower.eval()
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
                emb = model.client_tower(cc, cn, cm).cpu().numpy().astype("float32")
                embs.append(emb)
        client_emb_matrix = np.concatenate(embs, axis=0)
        out_df = pd.DataFrame({"client_id": client_df["client_id"].values})
        out_df["embedding"] = [row.tolist() for row in client_emb_matrix]
        out_df.to_parquet(client_emb_uri, index=False)
        print(f"Wrote client embeddings to: {client_emb_uri}")

        print("[train_and_log] Done.")
        runlog.write(f"FINISH ok=true elapsed_s={time.time() - t_start:.1f}")
    except Exception as e:
        runlog.write(f"FINISH ok=false elapsed_s={time.time() - t_start:.1f} error={e!r}")
        raise
