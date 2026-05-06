from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import torch


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _write_uri_bytes(uri: str, data: bytes) -> None:
    if uri.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem()
        with fs.open(uri, "wb") as f:
            f.write(data)
    else:
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _load_user_tower_model_from_state(state: dict):
    from two_tower.model.two_tower import DCNv2UserTower, UserMLPTower

    emb_dim = int(state["emb_dim"])
    expected_vocab_sizes = list(state.get("user_vocab_sizes", []))
    expected_num_dim = int(state.get("user_num_dim", 0))
    user_multi_vocab_sizes = list(state.get("user_multi_vocab_sizes") or [])
    user_multi_emb_dims = list(state.get("user_multi_emb_dims") or [])
    multi_cat_pool = str(state.get("multi_cat_pool", "mean"))
    use_pretrained_cat = bool(state.get("use_pretrained_cat", False))
    hidden = state.get("user_hidden")
    if hidden is None:
        hidden = state.get("user_deep_hidden")
    hidden = list(hidden or [512, 512])

    common_kwargs = dict(
        user_vocab_sizes=expected_vocab_sizes,
        user_num_dim=expected_num_dim,
        emb_dim=emb_dim,
        user_multi_vocab_sizes=user_multi_vocab_sizes or None,
        user_multi_emb_dims=user_multi_emb_dims or None,
        multi_pool=multi_cat_pool,
        use_pretrained_cat=use_pretrained_cat,
        pretrained_emb_dim=int(state.get("pretrained_emb_dim", 128)),
        target_cat_emb_dim=int(state.get("target_cat_emb_dim", 64)),
        freeze_base=bool(state.get("freeze_base", True)),
    )
    arch = str(state.get("user_tower_arch", "dcnv2")).lower()
    if arch == "mlp":
        model = UserMLPTower(hidden=hidden, **common_kwargs)
    else:
        model = DCNv2UserTower(
            num_cross_layers=int(state.get("num_cross_layers", 3)),
            deep_hidden=hidden,
            **common_kwargs,
        )
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def main() -> int:
    _ensure_src_on_path()

    from two_tower.config_loader import load_infer_job_config
    from two_tower.inference.artifact_paths import training_artifact_uris
    from two_tower.io.paths import artifact_uri
    from two_tower.io.uris import read_uri_bytes

    p = argparse.ArgumentParser(description="Export saved user tower checkpoint (.pt) to ONNX for Batch C backend.")
    p.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parents[1] / "configs" / "infer.yaml")),
        help="Path to configs/infer.yaml (used to resolve artifacts_base).",
    )
    p.add_argument(
        "--user-tower-uri",
        default=None,
        help="Optional explicit URI/path for user_tower_state.pt. Overrides artifacts_base lookup.",
    )
    p.add_argument(
        "--output-uri",
        default=None,
        help=(
            "Output URI/path for ONNX model. "
            "Defaults to <artifacts_base>/artifacts/user_tower/user_tower.onnx"
        ),
    )
    p.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = p.parse_args()

    infer_cfg = load_infer_job_config(args.config)
    artifacts_base = infer_cfg.paths.artifacts_base
    arts = training_artifact_uris(artifacts_base)
    user_tower_uri = args.user_tower_uri or arts["user_tower"]
    output_uri = args.output_uri or artifact_uri(artifacts_base, "artifacts", "user_tower", "user_tower.onnx")

    raw = read_uri_bytes(user_tower_uri)
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    model = _load_user_tower_model_from_state(state)

    batch_size = max(1, int(args.batch_size))
    user_cat_dim = len(list(state.get("user_vocab_sizes", [])))
    user_num_dim = int(state.get("user_num_dim", 0))
    user_multi_cols = len(list(state.get("user_multi_vocab_sizes") or []))
    max_tokens = int(state.get("multi_cat_max_tokens", 32))
    if max_tokens <= 0:
        max_tokens = 1

    user_cat = torch.zeros((batch_size, user_cat_dim), dtype=torch.long)
    user_num = torch.zeros((batch_size, user_num_dim), dtype=torch.float32)
    user_multi = torch.zeros((batch_size, user_multi_cols, max_tokens), dtype=torch.long)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.onnx.export(
            model,
            (user_cat, user_num, user_multi),
            str(tmp_path),
            input_names=["user_cat", "user_num", "user_multi"],
            output_names=["user_emb"],
            dynamic_axes={
                "user_cat": {0: "batch_size"},
                "user_num": {0: "batch_size"},
                "user_multi": {0: "batch_size"},
                "user_emb": {0: "batch_size"},
            },
            opset_version=int(args.opset),
            do_constant_folding=True,
        )
        _write_uri_bytes(output_uri, tmp_path.read_bytes())
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"[export_user_tower_onnx] source={user_tower_uri}")
    print(f"[export_user_tower_onnx] output={output_uri}")
    print(f"[export_user_tower_onnx] shape user_cat=(*,{user_cat_dim}) user_num=(*,{user_num_dim}) user_multi=(*,{user_multi_cols},{max_tokens})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

