from __future__ import annotations

import gc
import io
import os
import re
import time
import traceback
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pads
import torch
from torch import nn

from two_tower.features.encode import encode_cats, encode_multi_matrix, encode_nums
from two_tower.features.vocab import vocab_from_dict
from two_tower.io.uris import read_uri_bytes
from two_tower.model.two_tower import DCNv2UserTower


def _iter_record_batches(dset: Any, cols: list[str], batch_size: int):
    try:
        yield from pads.Scanner.from_dataset(dset, columns=cols, batch_size=batch_size).to_batches()
        return
    except Exception:
        pass
    for frag in dset.get_fragments():
        yield from frag.to_batches(batch_size=batch_size, columns=cols)


def _prepare_frame(
    pdf: pd.DataFrame,
    user_cat_cols: list[str],
    user_num_cols: list[str],
    user_multi_cols: list[str],
) -> pd.DataFrame:
    for c in user_cat_cols:
        if c not in pdf.columns:
            pdf[c] = "__NA__"
    for c in user_num_cols:
        if c not in pdf.columns:
            pdf[c] = 0.0
    for c in user_multi_cols:
        if c not in pdf.columns:
            pdf[c] = None
    return pdf


def _rank_batch_topk(
    *,
    infer_df: pd.DataFrame,
    device_id_col: str,
    user_cat_cols: list[str],
    user_num_cols: list[str],
    user_multi_cols: list[str],
    user_vocabs: dict,
    user_multi_vocabs: dict,
    multi_max_tokens: int,
    expected_vocab_sizes: list[int],
    expected_cat_dim: int,
    expected_num_dim: int,
    user_tower: nn.Module,
    client_emb_t: torch.Tensor,
    client_ids_np: np.ndarray,
    topk: int,
    client_chunk: int,
    use_amp: bool,
    amp_dtype_torch: torch.dtype,
    to_device: torch.device,
) -> pd.DataFrame:
    device_ids = infer_df[device_id_col].to_numpy()
    uc = encode_cats(infer_df, user_cat_cols, user_vocabs)
    un = encode_nums(infer_df, user_num_cols)
    um = encode_multi_matrix(infer_df, user_multi_cols, user_multi_vocabs, multi_max_tokens)

    if uc.shape[1] < expected_cat_dim:
        uc = torch.cat([uc, torch.zeros((uc.shape[0], expected_cat_dim - uc.shape[1]), dtype=uc.dtype)], dim=1)
    elif uc.shape[1] > expected_cat_dim:
        uc = uc[:, :expected_cat_dim]
    if un.shape[1] < expected_num_dim:
        un = torch.cat([un, torch.zeros((un.shape[0], expected_num_dim - un.shape[1]), dtype=un.dtype)], dim=1)
    elif un.shape[1] > expected_num_dim:
        un = un[:, :expected_num_dim]
    for i, vmax in enumerate(expected_vocab_sizes):
        if i < uc.shape[1]:
            uc[:, i] = uc[:, i].clamp(0, int(vmax) - 1)

    uc = uc.to(to_device)
    un = un.to(to_device)
    um = um.to(to_device)

    bsz = len(infer_df)
    n_clients = int(client_emb_t.shape[0])

    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype_torch)
        if to_device.type == "cuda" and use_amp
        else nullcontext()
    )

    with torch.inference_mode(), ctx:
        uemb = user_tower(uc, un, um)

        best_scores = torch.full((bsz, topk), -1.0, device=to_device, dtype=torch.float32)
        best_idx = torch.full((bsz, topk), -1, device=to_device, dtype=torch.int64)

        for start in range(0, n_clients, client_chunk):
            end = min(start + client_chunk, n_clients)
            chunk = client_emb_t[start:end]
            scores = torch.sigmoid((uemb @ chunk.T).float())
            k = min(topk, end - start)
            cs, ci = torch.topk(scores, k=k, dim=1)
            ci = ci + start
            ms = torch.cat([best_scores, cs], dim=1)
            mi = torch.cat([best_idx, ci], dim=1)
            best_scores, sel = torch.topk(ms, k=topk, dim=1)
            best_idx = torch.gather(mi, 1, sel)
            del chunk, scores, cs, ci, ms, mi, sel

    s_np = best_scores.detach().cpu().numpy().astype(np.float32)
    i_np = best_idx.detach().cpu().numpy().astype(np.int64)
    if to_device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(
        {
            device_id_col: np.repeat(device_ids, topk),
            "client_id": client_ids_np[i_np.reshape(-1)],
            "score": s_np.reshape(-1),
            "rank": np.tile(np.arange(1, topk + 1, dtype=np.int32), bsz),
        }
    )


def tt_infer_worker(
    worker_id: int,
    file_queue: Any,
    status_queue: Any,
    user_tower_uri: str,
    client_embeddings_uri: str,
    infer_ranking_output: str,
    device_id_col: str,
    user_cat_cols: list[str],
    user_num_cols: list[str],
    user_multi_cols: list[str],
    user_vocabs_raw: dict[str, dict],
    user_multi_vocabs_raw: dict[str, dict],
    multi_max_tokens: int,
    rank_user_batch: int,
    topk_clients: int,
    client_chunk: int,
    use_amp: bool,
    amp_dtype_str: str,
    output_min_rows: int,
    output_compression: str,
    infer_stream_batch_rows: int,
    workers_per_gpu: int,
    max_users_per_file: int | None,
) -> None:
    """Multiprocessing worker: load tower + client matrix once, consume parquet paths from ``file_queue``."""
    os.environ["POLARS_MAX_THREADS"] = "2"
    try:
        pl.Config.set_num_threads(2)
    except Exception:
        pass

    gpu_id = worker_id // max(workers_per_gpu, 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    to_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype_torch = getattr(torch, amp_dtype_str, torch.float16)

    user_vocabs = {k: vocab_from_dict(v) for k, v in user_vocabs_raw.items()}
    user_multi_vocabs = {k: vocab_from_dict(v) for k, v in user_multi_vocabs_raw.items()}

    t_load = time.time()
    raw = read_uri_bytes(user_tower_uri)
    _buf = io.BytesIO(raw)
    try:
        state = torch.load(_buf, map_location="cpu", weights_only=False)
    except TypeError:
        _buf.seek(0)
        state = torch.load(_buf, map_location="cpu")

    emb_dim = int(state["emb_dim"])
    expected_vocab_sizes = list(state.get("user_vocab_sizes", []))
    expected_cat_dim = len(expected_vocab_sizes)
    expected_num_dim = int(state.get("user_num_dim", 0))
    um = state.get("user_multi_vocab_sizes") or []
    umd = state.get("user_multi_emb_dims") or []
    mp = state.get("multi_cat_pool", "mean")
    use_pt_cat = bool(state.get("use_pretrained_cat", False))
    num_cross_layers = int(state.get("num_cross_layers", 3))
    deep_hidden = state.get("user_deep_hidden") or [512, 512]

    user_tower = DCNv2UserTower(
        user_vocab_sizes=expected_vocab_sizes,
        user_num_dim=expected_num_dim,
        emb_dim=emb_dim,
        num_cross_layers=num_cross_layers,
        deep_hidden=list(deep_hidden),
        user_multi_vocab_sizes=um if um else None,
        user_multi_emb_dims=umd if umd else None,
        multi_pool=mp,
        use_pretrained_cat=use_pt_cat,
        pretrained_emb_dim=int(state.get("pretrained_emb_dim", 128)),
        target_cat_emb_dim=int(state.get("target_cat_emb_dim", 64)),
        freeze_base=bool(state.get("freeze_base", True)),
    ).to(to_device)
    user_tower.load_state_dict(state["state_dict"])
    user_tower.eval()
    if hasattr(torch, "compile") and to_device.type == "cuda":
        try:
            user_tower = torch.compile(user_tower, mode="reduce-overhead")
        except Exception:
            pass
    del state
    gc.collect()

    client_emb_pl = pl.read_parquet(client_embeddings_uri)
    client_emb_pl = client_emb_pl.unique(subset=["client_id"], keep="first", maintain_order=False)
    client_ids_np = client_emb_pl["client_id"].to_numpy(allow_copy=True)
    emb_list = client_emb_pl["embedding"].to_list()
    del client_emb_pl
    client_emb_np = np.stack([np.asarray(e, dtype=np.float32) for e in emb_list], axis=0)
    del emb_list
    client_emb_t = torch.from_numpy(client_emb_np).to(to_device)
    del client_emb_np
    gc.collect()

    n_clients = int(client_emb_t.shape[0])
    topk = min(int(topk_clients), n_clients)

    print(
        f"  Worker {worker_id} -> GPU {gpu_id} | clients={n_clients} emb_dim={emb_dim} | load={time.time() - t_load:.1f}s",
        flush=True,
    )

    part = 0
    out_buf: list[pd.DataFrame] = []
    out_rows = 0
    out_prefix = infer_ranking_output

    def _flush(force: bool = False) -> None:
        nonlocal part, out_buf, out_rows
        if not out_buf or (not force and out_rows < output_min_rows):
            return
        merged = pl.concat([pl.from_pandas(d) for d in out_buf])
        path = f"{out_prefix}worker{worker_id:02d}_part_{part:06d}.parquet"
        merged.write_parquet(path, compression=output_compression)
        print(f"  [worker {worker_id}] wrote {path} ({len(merged):,} rows)", flush=True)
        del merged
        out_buf.clear()
        out_rows = 0
        part += 1

    while True:
        item = file_queue.get()
        if item is None:
            break
        parquet_path = item
        t_file = time.time()
        try:
            t_read = 0.0
            t_prep = 0.0
            t_inf = 0.0
            users_this_file = 0

            t0 = time.time()
            dset = pads.dataset(parquet_path, format="parquet")
            want = [device_id_col] + list(user_cat_cols) + list(user_num_cols) + list(user_multi_cols)
            cols = [c for c in want if c in dset.schema.names]

            seen: set[str] = set()
            pending: pl.DataFrame | None = None

            for batch in _iter_record_batches(dset, cols, infer_stream_batch_rows):
                if max_users_per_file is not None and users_this_file >= int(max_users_per_file):
                    break
                t_read += time.time() - t0
                t0 = time.time()
                bpl = pl.from_arrow(batch)
                bpl = bpl.unique(subset=[device_id_col], keep="first", maintain_order=False)
                bpl = bpl.filter(~pl.col(device_id_col).cast(pl.Utf8).is_in(seen))
                if bpl.is_empty():
                    t0 = time.time()
                    continue
                seen.update(bpl[device_id_col].cast(pl.Utf8).to_list())
                pending = pl.concat([pending, bpl], rechunk=False) if pending is not None else bpl
                del bpl
                t_prep += time.time() - t0

                while pending is not None and len(pending) >= rank_user_batch:
                    if max_users_per_file is not None and users_this_file >= int(max_users_per_file):
                        break
                    t0 = time.time()
                    chunk = _prepare_frame(
                        pending[:rank_user_batch].to_pandas(),
                        user_cat_cols,
                        user_num_cols,
                        user_multi_cols,
                    )
                    pending = pending[rank_user_batch:]
                    t_prep += time.time() - t0

                    if max_users_per_file is not None:
                        remaining = int(max_users_per_file) - users_this_file
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining].copy()

                    t0 = time.time()
                    out_df = _rank_batch_topk(
                        infer_df=chunk,
                        device_id_col=device_id_col,
                        user_cat_cols=user_cat_cols,
                        user_num_cols=user_num_cols,
                        user_multi_cols=user_multi_cols,
                        user_vocabs=user_vocabs,
                        user_multi_vocabs=user_multi_vocabs,
                        multi_max_tokens=multi_max_tokens,
                        expected_vocab_sizes=expected_vocab_sizes,
                        expected_cat_dim=expected_cat_dim,
                        expected_num_dim=expected_num_dim,
                        user_tower=user_tower,
                        client_emb_t=client_emb_t,
                        client_ids_np=client_ids_np,
                        topk=topk,
                        client_chunk=client_chunk,
                        use_amp=use_amp,
                        amp_dtype_torch=amp_dtype_torch,
                        to_device=to_device,
                    )
                    t_inf += time.time() - t0
                    out_buf.append(out_df)
                    out_rows += len(out_df)
                    users_this_file += len(chunk)
                    _flush()
                    del out_df
                    gc.collect()
                t0 = time.time()
                if max_users_per_file is not None and users_this_file >= int(max_users_per_file):
                    break

            if (
                pending is not None
                and len(pending) > 0
                and (max_users_per_file is None or users_this_file < int(max_users_per_file))
            ):
                t0 = time.time()
                chunk = _prepare_frame(
                    pending.to_pandas(),
                    user_cat_cols,
                    user_num_cols,
                    user_multi_cols,
                )
                del pending
                t_prep += time.time() - t0

                if max_users_per_file is not None:
                    remaining = int(max_users_per_file) - users_this_file
                    if remaining > 0 and len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining].copy()
                    elif remaining <= 0:
                        chunk = chunk.iloc[:0].copy()

                t0 = time.time()
                if len(chunk) > 0:
                    out_df = _rank_batch_topk(
                        infer_df=chunk,
                        device_id_col=device_id_col,
                        user_cat_cols=user_cat_cols,
                        user_num_cols=user_num_cols,
                        user_multi_cols=user_multi_cols,
                        user_vocabs=user_vocabs,
                        user_multi_vocabs=user_multi_vocabs,
                        multi_max_tokens=multi_max_tokens,
                        expected_vocab_sizes=expected_vocab_sizes,
                        expected_cat_dim=expected_cat_dim,
                        expected_num_dim=expected_num_dim,
                        user_tower=user_tower,
                        client_emb_t=client_emb_t,
                        client_ids_np=client_ids_np,
                        topk=topk,
                        client_chunk=client_chunk,
                        use_amp=use_amp,
                        amp_dtype_torch=amp_dtype_torch,
                        to_device=to_device,
                    )
                    t_inf += time.time() - t0
                    out_buf.append(out_df)
                    out_rows += len(out_df)
                    users_this_file += len(chunk)
                    _flush()
                    del out_df
                    gc.collect()

            status_queue.put(
                {
                    "worker": worker_id,
                    "file": parquet_path,
                    "users": users_this_file,
                    "read_time": t_read,
                    "preprocess_time": t_prep,
                    "inference_time": t_inf,
                    "total_time": time.time() - t_file,
                    "error": None,
                }
            )
        except Exception as e:
            status_queue.put(
                {
                    "worker": worker_id,
                    "file": parquet_path,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )

    _flush(force=True)

    del client_emb_t, client_ids_np
    if to_device.type == "cuda":
        torch.cuda.empty_cache()
    user_tower.cpu()
    del user_tower
    gc.collect()
    print(f"  Worker {worker_id} finished. Total parts written: {part}", flush=True)
