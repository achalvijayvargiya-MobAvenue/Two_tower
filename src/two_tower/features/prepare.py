from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np

from two_tower.configs import PipelineConfig
from two_tower.features.hash_embedding import build_all_hash_weights
from two_tower.features.schema import intersect_feature_cols, split_cat_num_multi
from two_tower.features.vocab import CatVocab, build_cat_vocab, build_multi_token_vocab


@dataclass
class FeatureArtifacts:
    """Everything needed to build datasets / models after Parquet load (reference notebook)."""

    user_feature_cols: list[str]
    client_feature_cols: list[str]
    user_multi_cols: list[str]
    client_multi_cols: list[str]
    user_cat_cols: list[str]
    user_num_cols: list[str]
    client_cat_cols: list[str]
    client_num_cols: list[str]
    user_vocabs: dict[str, CatVocab]
    client_vocabs: dict[str, CatVocab]
    user_multi_vocabs: dict[str, CatVocab]
    client_multi_vocabs: dict[str, CatVocab]
    user_cat_pretrained_weights: dict[str, np.ndarray] = field(default_factory=dict)
    client_cat_pretrained_weights: dict[str, np.ndarray] = field(default_factory=dict)


def prepare_training_features(
    train_df,
    val_df,
    cfg: PipelineConfig,
    *,
    build_hash_weights: bool = True,
) -> FeatureArtifacts:
    """Intersect feature lists, split cat/num/multi, fit vocabs on train, optional hash embedding tables."""
    fc = cfg.features
    tc = cfg.train

    if fc.label_col not in train_df.columns:
        raise KeyError(
            f"Train data missing label column {fc.label_col!r}. "
            f"Columns: {list(train_df.columns)[:40]} ..."
        )

    user_cols = list(fc.user_feature_cols)
    client_cols = list(fc.client_feature_cols)
    user_cols = intersect_feature_cols(train_df, user_cols, "USER")
    user_cols = intersect_feature_cols(val_df, user_cols, "USER (val)")
    client_cols = intersect_feature_cols(train_df, client_cols, "CLIENT")
    client_cols = intersect_feature_cols(val_df, client_cols, "CLIENT (val)")

    user_multi_cfg = list(fc.user_multi_cols)
    client_multi_cfg = list(fc.client_multi_cols)
    user_multi = [c for c in user_multi_cfg if c in train_df.columns]
    client_multi = [c for c in client_multi_cfg if c in train_df.columns]
    _missing_um = [c for c in user_multi_cfg if c not in train_df.columns]
    _missing_cm = [c for c in client_multi_cfg if c not in train_df.columns]
    if _missing_um:
        print("USER_MULTI: skipping (not in train):", _missing_um)
    if _missing_cm:
        print("CLIENT_MULTI: skipping (not in train):", _missing_cm)

    user_cat, user_num, _ = split_cat_num_multi(train_df, user_cols, set(user_multi))
    client_cat, client_num, _ = split_cat_num_multi(train_df, client_cols, set(client_multi))

    print("User cats:", len(user_cat), user_cat[:5])
    print("User nums:", len(user_num), user_num[:5])
    print("User multi:", user_multi)
    print("Client cats:", len(client_cat), client_cat)
    print("Client nums:", len(client_num), client_num[:5])
    print("Client multi:", client_multi)

    min_c = tc.min_count
    n_oov = tc.num_oov_buckets
    user_vocabs = {c: build_cat_vocab(train_df[c], min_c, n_oov) for c in user_cat}
    client_vocabs = {c: build_cat_vocab(train_df[c], min_c, n_oov) for c in client_cat}
    user_multi_vocabs = {c: build_multi_token_vocab(train_df[c], min_c, n_oov) for c in user_multi}
    client_multi_vocabs = {c: build_multi_token_vocab(train_df[c], min_c, n_oov) for c in client_multi}

    print("User vocab sizes:", {c: v.size for c, v in user_vocabs.items()})
    print("Client vocab sizes:", {c: v.size for c, v in client_vocabs.items()})
    if user_multi_vocabs:
        print("User multi vocab sizes:", {c: v.size for c, v in user_multi_vocabs.items()})
    if client_multi_vocabs:
        print("Client multi vocab sizes:", {c: v.size for c, v in client_multi_vocabs.items()})

    user_hw: dict[str, np.ndarray] = {}
    client_hw: dict[str, np.ndarray] = {}
    if build_hash_weights:
        pre = int(tc.pretrained_emb_dim)
        if user_vocabs:
            print("Building hash embeddings for user categorical features ...")
            user_hw = build_all_hash_weights(user_vocabs, pre)
        if client_vocabs:
            print("Building hash embeddings for client categorical features ...")
            client_hw = build_all_hash_weights(client_vocabs, pre)
        gc.collect()
        _tu = sum(v.n_frequent for v in user_vocabs.values())
        _tc = sum(v.n_frequent for v in client_vocabs.values())
        print(
            f"Hash tables: user {len(user_hw)} cols / {_tu} frequent tokens; "
            f"client {len(client_hw)} cols / {_tc} frequent tokens; dim={pre}."
        )

    return FeatureArtifacts(
        user_feature_cols=user_cols,
        client_feature_cols=client_cols,
        user_multi_cols=user_multi,
        client_multi_cols=client_multi,
        user_cat_cols=user_cat,
        user_num_cols=user_num,
        client_cat_cols=client_cat,
        client_num_cols=client_num,
        user_vocabs=user_vocabs,
        client_vocabs=client_vocabs,
        user_multi_vocabs=user_multi_vocabs,
        client_multi_vocabs=client_multi_vocabs,
        user_cat_pretrained_weights=user_hw,
        client_cat_pretrained_weights=client_hw,
    )
