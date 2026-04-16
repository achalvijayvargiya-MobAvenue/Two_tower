from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

from two_tower.configs import InferJobConfig
from two_tower.inference.artifact_paths import load_vocab_artifact_pickle, training_artifact_uris
from two_tower.inference.list_inputs import list_parquet_inputs
from two_tower.inference.worker import tt_infer_worker


def run_inference_job(cfg: InferJobConfig) -> None:
    """Spawn workers to rank users against precomputed client embeddings (reference flow)."""
    ic = cfg.infer
    if ic.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    arts = training_artifact_uris(cfg.paths.artifacts_base)
    vocab = load_vocab_artifact_pickle(arts["vocab"])

    user_vocabs_raw = vocab["user_vocabs"]
    user_multi_vocabs_raw = vocab["user_multi_vocabs"]
    user_cat_cols = list(vocab["user_cat_cols"])
    user_num_cols = list(vocab["user_num_cols"])
    user_multi_cols = list(vocab["user_multi_cols"])
    device_id_col = str(vocab["device_id_col"])
    multi_max_tokens = int(vocab["multi_cat_max_tokens"])

    infer_files = list_parquet_inputs(cfg.paths.infer)
    if not infer_files:
        raise FileNotFoundError(f"No inference files under {cfg.paths.infer!r}")

    out_dir = ic.ranking_output.rstrip("/")
    if not out_dir.startswith("s3://"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir if out_dir.endswith("/") else out_dir + "/"

    num_workers = max(1, int(ic.num_physical_gpus) * max(1, int(ic.workers_per_gpu)))

    print(f"[infer] {num_workers} workers | {len(infer_files)} files | top-{ic.topk_clients}")
    print(f"[infer] user tower: {arts['user_tower']}")
    print(f"[infer] client emb: {arts['client_embeddings']}")
    print(f"[infer] output prefix: {out_prefix}")

    ctx = multiprocessing.get_context("spawn")
    file_queue = ctx.Queue()
    status_queue = ctx.Queue()

    for fk in infer_files:
        file_queue.put(fk)
    for _ in range(num_workers):
        file_queue.put(None)

    start_time = time.time()
    processes: list[multiprocessing.Process] = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=tt_infer_worker,
            args=(
                worker_id,
                file_queue,
                status_queue,
                arts["user_tower"],
                arts["client_embeddings"],
                out_prefix,
                device_id_col,
                user_cat_cols,
                user_num_cols,
                user_multi_cols,
                user_vocabs_raw,
                user_multi_vocabs_raw,
                multi_max_tokens,
                ic.rank_user_batch,
                ic.topk_clients,
                ic.client_chunk,
                ic.use_amp,
                ic.amp_dtype,
                ic.output_min_rows_per_part,
                ic.output_parquet_compression,
                ic.infer_stream_batch_rows,
                ic.workers_per_gpu,
            ),
        )
        p.start()
        processes.append(p)

    completed = 0
    total = len(infer_files)
    errors: list[dict] = []
    total_read = total_prep = total_inf = 0.0

    while completed < total:
        try:
            status = status_queue.get(timeout=3600)
        except Exception:
            print("[infer] status queue timeout; stopping progress loop.")
            break
        completed += 1
        if status.get("error"):
            errors.append(status)
            print(f"[ERROR] worker={status.get('worker')} file={status.get('file')}")
            print(f"        {status.get('error')}")
        else:
            total_read += float(status.get("read_time", 0))
            total_prep += float(status.get("preprocess_time", 0))
            total_inf += float(status.get("inference_time", 0))
            if completed % 25 == 0 or completed == total:
                elapsed = (time.time() - start_time) / 60
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{total} | ~{rate:.1f} files/min")

    for p in processes:
        p.join()

    elapsed_m = (time.time() - start_time) / 60
    print(
        f"Inference finished in {elapsed_m:.1f} min | errors={len(errors)} | "
        f"cumulative read/prep/inf (s): {total_read:.1f} / {total_prep:.1f} / {total_inf:.1f}"
    )
    if errors:
        raise RuntimeError(f"Inference had {len(errors)} failed file(s); see logs above.")
