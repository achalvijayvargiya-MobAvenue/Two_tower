from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (train).")
    args = p.parse_args()

    # We’ll implement config loading next (yaml -> PipelineConfig).
    from two_tower.training import train_and_log

    train_and_log(cfg=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

