from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()

    p = argparse.ArgumentParser(description="Two-tower batch inference (user-only parquet -> rankings).")
    p.add_argument("--config", required=True, help="Path to configs/infer.yaml")
    args = p.parse_args()

    from two_tower.config_loader import load_infer_job_config
    from two_tower.inference.run import run_inference_job

    cfg = load_infer_job_config(args.config)
    run_inference_job(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
