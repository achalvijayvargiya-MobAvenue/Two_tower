from __future__ import annotations

import hashlib

import numpy as np

from two_tower.features.vocab import CatVocab


def hash_embed_token(token: str, dim: int) -> np.ndarray:
    """Map a string to a unit-normed float32 vector (reference: SHA-256 multi-seed)."""
    v = np.empty(dim, dtype=np.float32)
    for seed in range((dim + 3) // 4):
        h = int(hashlib.sha256(f"{seed}:{token}".encode()).hexdigest(), 16)
        for i in range(4):
            idx = seed * 4 + i
            if idx < dim:
                v[idx] = ((h >> (i * 16)) & 0xFFFF) / 32768.0 - 1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else v


def build_hash_weight_matrix(vocab: CatVocab, pre_dim: int) -> np.ndarray:
    """(vocab.size × pre_dim) for nn.Embedding init (row 0 = padding; OOV rows stay zero)."""
    id_to_token = {v: k for k, v in vocab.token_to_id.items()}
    w = np.zeros((vocab.size, pre_dim), dtype=np.float32)
    for _idx, tok in id_to_token.items():
        w[_idx] = hash_embed_token(tok, pre_dim)
    return w


def build_all_hash_weights(vocabs: dict[str, CatVocab], pre_dim: int) -> dict[str, np.ndarray]:
    return {col: build_hash_weight_matrix(v, pre_dim) for col, v in vocabs.items()}
