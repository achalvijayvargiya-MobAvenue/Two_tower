from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _must_s3(uri: str, label: str) -> str:
    u = str(uri).strip()
    if not u.startswith("s3://"):
        raise ValueError(f"{label} is not an s3 path: {u!r}")
    return u.rstrip("/") + "/"


def _resolve_aws_cli() -> str:
    # Prefer PATH lookup first.
    for name in ("aws", "aws.exe", "aws.cmd"):
        p = shutil.which(name)
        if p:
            return p

    # On Windows, aws.cmd is often installed in Python's Scripts directory.
    py_scripts = Path(sys.executable).resolve().parent / "Scripts"
    for name in ("aws.cmd", "aws.exe"):
        cand = py_scripts / name
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        "Could not find AWS CLI executable. Install AWS CLI or ensure aws/aws.cmd is on PATH."
    )


def _aws_rm_recursive(aws_cli: str, uri: str, dry_run: bool) -> None:
    cmd = [aws_cli, "s3", "rm", uri, "--recursive"]
    if dry_run:
        cmd.append("--dryrun")
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clean configured S3 output paths using aws s3 rm --recursive. "
            "Reads infer.ranking_output from configs/infer.yaml and "
            "paths.artifacts_base from configs/train.yaml."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete objects. By default the script runs in --dryrun mode.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    infer_cfg = _load_yaml(repo_root / "configs" / "infer.yaml")
    train_cfg = _load_yaml(repo_root / "configs" / "train.yaml")

    ranking_output = _must_s3(infer_cfg["infer"]["ranking_output"], "infer.ranking_output")
    artifacts_base = _must_s3(train_cfg["paths"]["artifacts_base"], "paths.artifacts_base")

    paths = []
    for p in (ranking_output, artifacts_base):
        if p not in paths:
            paths.append(p)

    dry_run = not args.execute
    mode = "DRY RUN" if dry_run else "EXECUTE"
    aws_cli = _resolve_aws_cli()
    print(f"[clean_s3_paths] mode={mode}")
    print(f"[clean_s3_paths] aws_cli={aws_cli}")
    for p in paths:
        _aws_rm_recursive(aws_cli, p, dry_run=dry_run)

    print("[clean_s3_paths] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

