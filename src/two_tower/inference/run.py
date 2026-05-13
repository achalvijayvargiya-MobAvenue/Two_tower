from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

from two_tower.configs import InferJobConfig
from two_tower.inference.artifact_paths import load_vocab_artifact_pickle, training_artifact_uris
from two_tower.inference.list_inputs import list_parquet_inputs
from two_tower.inference.worker import tt_infer_worker
from two_tower.io.runlog import start_run_log


def run_inference_job(cfg: InferJobConfig) -> None:
    """Spawn workers to rank users against precomputed client embeddings (reference flow)."""
    ic = cfg.infer
    runlog = start_run_log(kind="infer", name="two_tower")
    t_start = time.time()
    if ic.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if int(ic.pyarrow_io_threads) > 0:
        os.environ["PYARROW_IO_THREADS"] = str(int(ic.pyarrow_io_threads))

    arts = training_artifact_uris(cfg.paths.artifacts_base)
    vocab = load_vocab_artifact_pickle(arts["vocab"])

    user_vocabs_raw = vocab["user_vocabs"]
    user_multi_vocabs_raw = vocab["user_multi_vocabs"]
    user_cat_cols = list(vocab["user_cat_cols"])
    user_num_cols = list(vocab["user_num_cols"])
    user_multi_cols = list(vocab["user_multi_cols"])
    vocab_device_id_col = str(vocab["device_id_col"])
    device_id_candidates: list[str] = []
    if cfg.infer_parquet_device_id_col is not None:
        device_id_candidates.append(cfg.infer_parquet_device_id_col)
    device_id_candidates.append(vocab_device_id_col)
    for alt in ("device_id", "dev_id", "ifa", "advertising_id"):
        if alt not in device_id_candidates:
            device_id_candidates.append(alt)
    ranking_device_id_out = str(cfg.infer.ranking_device_id_col)
    multi_max_tokens = int(vocab["multi_cat_max_tokens"])

    infer_files = list_parquet_inputs(cfg.paths.infer)
    if not infer_files:
        raise FileNotFoundError(f"No inference files under {cfg.paths.infer!r}")
    if ic.max_files is not None:
        infer_files = infer_files[: max(0, int(ic.max_files))]
        if not infer_files:
            raise FileNotFoundError(
                f"infer.max_files={ic.max_files} filtered out all inputs under {cfg.paths.infer!r}"
            )

    out_dir = ic.ranking_output.rstrip("/")
    if not out_dir.startswith("s3://"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir if out_dir.endswith("/") else out_dir + "/"

    num_workers = max(1, int(ic.num_physical_gpus) * max(1, int(ic.workers_per_gpu)))

    print(
        f"[infer] {num_workers} workers | {len(infer_files)} files | top-{ic.topk_clients} | "
        f"device_id candidates={device_id_candidates} -> output col={ranking_device_id_out!r}"
    )
    if ic.max_files is not None or ic.max_users_per_file is not None:
        print(f"[infer] limits: max_files={ic.max_files} max_users_per_file={ic.max_users_per_file}")
    print(f"[infer] user tower: {arts['user_tower']}")
    print(f"[infer] client emb: {arts['client_embeddings']}")
    print(f"[infer] output prefix: {out_prefix}")
    runlog.write(
        f"CONFIG infer_path={cfg.paths.infer} artifacts_base={cfg.paths.artifacts_base} "
        f"out={out_prefix} files={len(infer_files)} topk={ic.topk_clients} "
        f"max_files={ic.max_files} max_users_per_file={ic.max_users_per_file} "
        f"device_id_candidates={device_id_candidates} ranking_device_id_col={ranking_device_id_out!r} "
        f"(worker device-id resolution lines are echoed here from the parent process; "
        f"child stdout may be hidden on SageMaker)"
    )

    # Jupyter/IPython notebooks + multiprocessing "spawn" can fail because the worker function
    # ends up living in `__main__` (not importable in child processes). On Linux, "fork"
    # avoids pickling the target and is the most compatible for notebooks.
    #
    # For CLI/prod-like execution, we keep the safer default of "spawn".
    start_method = "spawn"
    try:
        in_notebook = bool(getattr(__import__("builtins"), "get_ipython", None))
    except Exception:
        in_notebook = False
    if in_notebook:
        try:
            methods = multiprocessing.get_all_start_methods()
        except Exception:
            methods = []
        if "fork" in methods:
            start_method = "fork"
    ctx = multiprocessing.get_context(start_method)
    file_queue = ctx.Queue()
    status_queue = ctx.Queue()
    abort_event = ctx.Event()
    guard_lock = ctx.Lock()
    guard_stage = ctx.Value("i", 0)  # 0=unset, 1=passed, -1=failed (see worker guard)

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
                device_id_candidates,
                ranking_device_id_out,
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
                ic.max_users_per_file,
                ic.user_tower_backend,
                ic.user_tower_onnx_uri,
                ic.trt_fp16_enable,
                ic.trt_engine_cache_enable,
                ic.trt_engine_cache_path,
                abort_event,
                guard_lock,
                guard_stage,
                ic.device_id_output_guard_rows,
                ic.device_id_output_guard_stop_job,
                ic.pyarrow_io_threads,
                ic.gc_every_n_rank_batches,
            ),
        )
        p.start()
        processes.append(p)

    completed = 0
    total = len(infer_files)
    errors: list[dict] = []
    total_read = total_prep = total_inf = 0.0
    last_hb = time.time()
    hb_every_s = 60.0

    fatal_status: dict | None = None
    while completed < total:
        try:
            status = status_queue.get(timeout=30)
        except Exception:
            now = time.time()
            if now - last_hb >= hb_every_s:
                runlog.write(f"HEARTBEAT completed_files={completed}/{total} elapsed_min={(now - start_time)/60:.1f}")
                last_hb = now
            continue
        if status.get("fatal_abort"):
            fatal_status = status
            break
        completed += 1
        if status.get("error"):
            errors.append(status)
            print(f"[ERROR] worker={status.get('worker')} file={status.get('file')}")
            print(f"        {status.get('error')}")
            runlog.write(f"FILE_DONE ok=false file={status.get('file')} worker={status.get('worker')} error={status.get('error')}")
        else:
            total_read += float(status.get("read_time", 0))
            total_prep += float(status.get("preprocess_time", 0))
            total_inf += float(status.get("inference_time", 0))
            src_col = status.get("device_id_source_col")
            peek_nn = status.get("device_id_peek_nonnull")
            peek_nr = status.get("device_id_peek_rows")
            runlog.write(
                f"FILE_DONE ok=true file={status.get('file')} worker={status.get('worker')} users={status.get('users')} "
                f"t_read={status.get('read_time',0):.2f} t_prep={status.get('preprocess_time',0):.2f} "
                f"t_inf={status.get('inference_time',0):.2f} t_total={status.get('total_time',0):.2f} "
                f"device_id_source_col={src_col} device_id_peek_nonnull={peek_nn} device_id_peek_rows={peek_nr}"
            )
            if src_col is not None:
                warn = peek_nn == 0 or peek_nn is None
                if completed <= 5 or warn or completed % 200 == 0 or completed == total:
                    print(
                        f"[infer] file {completed}/{total} device_id_source_col={src_col!r} "
                        f"peek_nonnull={peek_nn} peek_rows={peek_nr}"
                        + ("  <<< WARNING: no non-null ids in peek batch" if warn else ""),
                        flush=True,
                    )
            if completed % 25 == 0 or completed == total:
                elapsed = (time.time() - start_time) / 60
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{total} | ~{rate:.1f} files/min")
        now = time.time()
        if now - last_hb >= hb_every_s:
            runlog.write(f"HEARTBEAT completed_files={completed}/{total} elapsed_min={(now - start_time)/60:.1f}")
            last_hb = now

    if fatal_status is not None:
        err = str(fatal_status.get("error", "fatal_abort"))
        runlog.write(f"FATAL_ABORT worker={fatal_status.get('worker')} file={fatal_status.get('file')} error={err}")
        print(f"[infer][FATAL_ABORT] {err}", flush=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=120)
        runlog.write(f"FINISH ok=false fatal_abort elapsed_s={time.time()-t_start:.1f}")
        raise RuntimeError(err)

    for p in processes:
        p.join()

    elapsed_m = (time.time() - start_time) / 60
    print(
        f"Inference finished in {elapsed_m:.1f} min | errors={len(errors)} | "
        f"cumulative read/prep/inf (s): {total_read:.1f} / {total_prep:.1f} / {total_inf:.1f}"
    )
    if errors:
        runlog.write(f"FINISH ok=false errors={len(errors)} elapsed_s={time.time()-t_start:.1f}")
        raise RuntimeError(f"Inference had {len(errors)} failed file(s); see logs above.")
    runlog.write(f"FINISH ok=true errors=0 elapsed_s={time.time()-t_start:.1f}")
