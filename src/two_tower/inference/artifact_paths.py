from __future__ import annotations

import pickle

from two_tower.io.paths import artifact_uri
from two_tower.io.uris import read_uri_bytes


def training_artifact_uris(artifacts_base: str) -> dict[str, str]:
    base = artifacts_base.rstrip("/")
    return {
        "user_tower": artifact_uri(base, "artifacts", "user_tower", "user_tower_state.pt"),
        "vocab": artifact_uri(base, "artifacts", "vocab_artifact", "vocab_artifact.pkl"),
        "client_embeddings": artifact_uri(base, "artifacts", "client_embeddings", "client_embeddings.parquet"),
    }


def load_vocab_artifact_pickle(uri: str) -> dict:
    return pickle.loads(read_uri_bytes(uri))
