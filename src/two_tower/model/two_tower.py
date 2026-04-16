from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def embedding_dim_for_cardinality(vocab_size: int) -> int:
    v = int(max(1, vocab_size))
    if v < 100:
        return 6
    if v < 10_000:
        return 20
    if v < 1_000_000:
        return 64
    return 128


class CatEmbedder(nn.Module):
    """Categorical embedder: random init or hash-pretrained + linear projection."""

    def __init__(
        self,
        vocab_sizes: List[int],
        use_pretrained: bool = False,
        pretrained_weights: list | None = None,
        pretrained_emb_dim: int = 384,
        target_emb_dim_per_col: int = 64,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.emb_dims = [target_emb_dim_per_col] * len(vocab_sizes)
            self.embs = nn.ModuleList([nn.Embedding(int(v), pretrained_emb_dim, padding_idx=0) for v in vocab_sizes])
            if pretrained_weights is not None:
                for emb, w_np in zip(self.embs, pretrained_weights):
                    w = torch.from_numpy(w_np).float()
                    n = min(w.shape[0], emb.weight.shape[0])
                    emb.weight.data[:n].copy_(w[:n])
            if freeze_base:
                for emb in self.embs:
                    emb.weight.requires_grad_(False)
            self.projs: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(pretrained_emb_dim, target_emb_dim_per_col, bias=False) for _ in vocab_sizes]
            )
        else:
            self.emb_dims = [embedding_dim_for_cardinality(v) for v in vocab_sizes]
            self.embs = nn.ModuleList([nn.Embedding(int(v), int(d)) for v, d in zip(vocab_sizes, self.emb_dims)])
            self.projs = None

    def output_dim(self) -> int:
        return int(sum(self.emb_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((x.shape[0], 0), device=x.device)
        if self.use_pretrained and self.projs is not None:
            parts = [proj(emb(x[:, i])) for i, (emb, proj) in enumerate(zip(self.embs, self.projs))]
        else:
            parts = [emb(x[:, i]) for i, emb in enumerate(self.embs)]
        return torch.cat(parts, dim=1)


class TokenSeqAttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(int(d), 1)

    def forward(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(e).squeeze(-1)
        s = s.masked_fill(mask <= 0, -1e9)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (e * w).sum(dim=1)


class PooledMultiCatEmbedder(nn.Module):
    def __init__(self, vocab_sizes: List[int], emb_dims: List[int], pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.emb_dims = [int(d) for d in emb_dims]
        self.embs = nn.ModuleList(
            [nn.Embedding(int(v), int(d), padding_idx=0) for v, d in zip(vocab_sizes, self.emb_dims)]
        )
        self._attn: nn.ModuleList | None = None
        if pool == "attention":
            self._attn = nn.ModuleList([TokenSeqAttentionPool(d) for d in self.emb_dims])

    def output_dim(self) -> int:
        return int(sum(self.emb_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros((x.shape[0], 0), device=x.device)
        outs = []
        for i, emb in enumerate(self.embs):
            ids = x[:, i, :]
            e = emb(ids)
            mask = (ids != 0).float()
            if self.pool == "sum":
                pooled = (e * mask.unsqueeze(-1)).sum(dim=1)
            elif self.pool == "attention" and self._attn is not None:
                pooled = self._attn[i](e, mask)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (e * mask.unsqueeze(-1)).sum(dim=1) / denom
            outs.append(pooled)
        return torch.cat(outs, dim=1)


def _safe_l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    denom = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


class DCNv2CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        cross = xl @ self.weight
        cross = cross + self.bias
        cross = x0 * cross
        return cross + xl


class DCNv2UserTower(nn.Module):
    def __init__(
        self,
        user_vocab_sizes: List[int],
        user_num_dim: int,
        emb_dim: int = 1024,
        num_cross_layers: int = 3,
        deep_hidden: List[int] | None = None,
        user_multi_vocab_sizes: List[int] | None = None,
        user_multi_emb_dims: List[int] | None = None,
        multi_pool: str = "mean",
        use_pretrained_cat: bool = False,
        user_cat_pretrained_weights: list | None = None,
        pretrained_emb_dim: int = 384,
        target_cat_emb_dim: int = 64,
        freeze_base: bool = False,
    ):
        super().__init__()
        if deep_hidden is None:
            deep_hidden = [512, 512]

        self.user_cat = CatEmbedder(
            user_vocab_sizes,
            use_pretrained=use_pretrained_cat,
            pretrained_weights=user_cat_pretrained_weights,
            pretrained_emb_dim=pretrained_emb_dim,
            target_emb_dim_per_col=target_cat_emb_dim,
            freeze_base=freeze_base,
        )
        ms = list(user_multi_vocab_sizes or [])
        if ms:
            ed = user_multi_emb_dims or [embedding_dim_for_cardinality(v) for v in ms]
            self.user_multi = PooledMultiCatEmbedder(ms, ed, pool=multi_pool)
        else:
            self.user_multi = None

        cat_w = self.user_cat.output_dim() + (self.user_multi.output_dim() if self.user_multi else 0)
        input_dim = int(cat_w + user_num_dim)

        self.cross_layers = nn.ModuleList([DCNv2CrossLayer(input_dim) for _ in range(num_cross_layers)])

        deep_layers: list[nn.Module] = []
        prev = input_dim
        for h in deep_hidden:
            deep_layers.append(nn.Linear(prev, h))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.BatchNorm1d(h))
            prev = h
        self.deep = nn.Sequential(*deep_layers)

        self.fc = nn.Linear(prev + input_dim, emb_dim)

    def forward(self, user_cat: torch.Tensor, user_num: torch.Tensor, user_multi: torch.Tensor) -> torch.Tensor:
        cat = self.user_cat(user_cat)
        if self.user_multi is not None:
            cat = torch.cat([cat, self.user_multi(user_multi)], dim=1)
        x0 = torch.cat([cat, user_num], dim=1)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        cross_out = xl
        deep_out = self.deep(x0)
        emb = self.fc(torch.cat([cross_out, deep_out], dim=1))
        return _safe_l2_normalize(emb, dim=1)


class ClientMLPTower(nn.Module):
    def __init__(
        self,
        client_vocab_sizes: List[int],
        client_num_dim: int,
        emb_dim: int = 1024,
        hidden: List[int] | None = None,
        client_multi_vocab_sizes: List[int] | None = None,
        client_multi_emb_dims: List[int] | None = None,
        multi_pool: str = "mean",
        use_pretrained_cat: bool = False,
        client_cat_pretrained_weights: list | None = None,
        pretrained_emb_dim: int = 384,
        target_cat_emb_dim: int = 64,
        freeze_base: bool = False,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

        self.client_cat = CatEmbedder(
            client_vocab_sizes,
            use_pretrained=use_pretrained_cat,
            pretrained_weights=client_cat_pretrained_weights,
            pretrained_emb_dim=pretrained_emb_dim,
            target_emb_dim_per_col=target_cat_emb_dim,
            freeze_base=freeze_base,
        )
        ms = list(client_multi_vocab_sizes or [])
        if ms:
            ed = client_multi_emb_dims or [embedding_dim_for_cardinality(v) for v in ms]
            self.client_multi = PooledMultiCatEmbedder(ms, ed, pool=multi_pool)
        else:
            self.client_multi = None

        cat_w = self.client_cat.output_dim() + (self.client_multi.output_dim() if self.client_multi else 0)
        input_dim = int(cat_w + client_num_dim)

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            prev = h
        layers.append(nn.Linear(prev, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, client_cat: torch.Tensor, client_num: torch.Tensor, client_multi: torch.Tensor) -> torch.Tensor:
        cat = self.client_cat(client_cat)
        if self.client_multi is not None:
            cat = torch.cat([cat, self.client_multi(client_multi)], dim=1)
        x = torch.cat([cat, client_num], dim=1)
        emb = self.net(x)
        return _safe_l2_normalize(emb, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_vocab_sizes: List[int],
        user_num_dim: int,
        client_vocab_sizes: List[int],
        client_num_dim: int,
        emb_dim: int = 1024,
        user_multi_vocab_sizes: List[int] | None = None,
        user_multi_emb_dims: List[int] | None = None,
        client_multi_vocab_sizes: List[int] | None = None,
        client_multi_emb_dims: List[int] | None = None,
        multi_pool: str = "mean",
        use_pretrained_cat: bool = False,
        user_cat_pretrained_weights: list | None = None,
        client_cat_pretrained_weights: list | None = None,
        pretrained_emb_dim: int = 384,
        target_cat_emb_dim: int = 64,
        freeze_pretrained_base: bool = False,
        num_cross_layers: int = 2,
        user_deep_hidden: List[int] | None = None,
        client_hidden: List[int] | None = None,
    ):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.user_tower = DCNv2UserTower(
            user_vocab_sizes,
            user_num_dim,
            emb_dim=emb_dim,
            num_cross_layers=num_cross_layers,
            deep_hidden=user_deep_hidden,
            user_multi_vocab_sizes=user_multi_vocab_sizes,
            user_multi_emb_dims=user_multi_emb_dims,
            multi_pool=multi_pool,
            use_pretrained_cat=use_pretrained_cat,
            user_cat_pretrained_weights=user_cat_pretrained_weights,
            pretrained_emb_dim=pretrained_emb_dim,
            target_cat_emb_dim=target_cat_emb_dim,
            freeze_base=freeze_pretrained_base,
        )
        self.client_tower = ClientMLPTower(
            client_vocab_sizes,
            client_num_dim,
            emb_dim=emb_dim,
            hidden=client_hidden,
            client_multi_vocab_sizes=client_multi_vocab_sizes,
            client_multi_emb_dims=client_multi_emb_dims,
            multi_pool=multi_pool,
            use_pretrained_cat=use_pretrained_cat,
            client_cat_pretrained_weights=client_cat_pretrained_weights,
            pretrained_emb_dim=pretrained_emb_dim,
            target_cat_emb_dim=target_cat_emb_dim,
            freeze_base=freeze_pretrained_base,
        )

    def forward(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        client_cat: torch.Tensor,
        client_num: torch.Tensor,
        user_multi: torch.Tensor,
        client_multi: torch.Tensor,
    ):
        user_emb = self.user_tower(user_cat, user_num, user_multi)
        client_emb = self.client_tower(client_cat, client_num, client_multi)
        scale = self.log_scale.clamp(math.log(1.0), math.log(100.0)).exp()
        logits = (user_emb * client_emb).sum(dim=1, keepdim=True) * scale
        return logits, user_emb, client_emb


def build_two_tower_model(fa, cfg) -> TwoTowerModel:
    """Construct model from feature artifacts + pipeline config."""
    from two_tower.configs import PipelineConfig
    from two_tower.features.prepare import FeatureArtifacts

    assert isinstance(cfg, PipelineConfig)
    assert isinstance(fa, FeatureArtifacts)

    tc = cfg.train
    user_vocab_sizes = [fa.user_vocabs[c].size for c in fa.user_cat_cols]
    client_vocab_sizes = [fa.client_vocabs[c].size for c in fa.client_cat_cols]
    user_multi_vocab_sizes = [fa.user_multi_vocabs[c].size for c in fa.user_multi_cols]
    client_multi_vocab_sizes = [fa.client_multi_vocabs[c].size for c in fa.client_multi_cols]
    user_multi_emb_dims = [embedding_dim_for_cardinality(v) for v in user_multi_vocab_sizes]
    client_multi_emb_dims = [embedding_dim_for_cardinality(v) for v in client_multi_vocab_sizes]

    user_num_dim = len(fa.user_num_cols)
    client_num_dim = len(fa.client_num_cols)

    use_pt = bool(fa.user_cat_pretrained_weights) and bool(fa.client_cat_pretrained_weights)
    user_w = [fa.user_cat_pretrained_weights[c] for c in fa.user_cat_cols] if use_pt and fa.user_cat_cols else None
    client_w = (
        [fa.client_cat_pretrained_weights[c] for c in fa.client_cat_cols] if use_pt and fa.client_cat_cols else None
    )

    return TwoTowerModel(
        user_vocab_sizes=user_vocab_sizes,
        user_num_dim=user_num_dim,
        client_vocab_sizes=client_vocab_sizes,
        client_num_dim=client_num_dim,
        emb_dim=tc.embed_dim,
        num_cross_layers=tc.dcn_cross_layers,
        user_deep_hidden=list(tc.mlp_hidden_dims),
        client_hidden=list(tc.client_mlp_hidden),
        user_multi_vocab_sizes=user_multi_vocab_sizes or None,
        user_multi_emb_dims=user_multi_emb_dims or None,
        client_multi_vocab_sizes=client_multi_vocab_sizes or None,
        client_multi_emb_dims=client_multi_emb_dims or None,
        multi_pool=tc.multi_cat_pool,
        use_pretrained_cat=use_pt,
        user_cat_pretrained_weights=user_w,
        client_cat_pretrained_weights=client_w,
        pretrained_emb_dim=tc.pretrained_emb_dim,
        target_cat_emb_dim=tc.pretrained_cat_emb_dim,
        freeze_pretrained_base=tc.freeze_pretrained_base,
    )
