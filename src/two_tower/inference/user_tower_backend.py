from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import nn


class UserTowerBackend(Protocol):
    """Runtime backend abstraction for user tower embedding inference."""

    backend_name: str

    def infer(self, user_cat: torch.Tensor, user_num: torch.Tensor, user_multi: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class TorchUserTowerBackend:
    """PyTorch backend (current default path)."""

    model: nn.Module
    backend_name: str = "pytorch"

    def infer(self, user_cat: torch.Tensor, user_num: torch.Tensor, user_multi: torch.Tensor) -> torch.Tensor:
        return self.model(user_cat, user_num, user_multi)


@dataclass
class OnnxRuntimeTensorRTUserTowerBackend:
    """
    TensorRT backend via ONNX Runtime provider stack.

    Provider order:
    1) TensorRTExecutionProvider
    2) CUDAExecutionProvider
    3) CPUExecutionProvider
    """

    session: object
    output_name: str
    device: torch.device
    backend_name: str = "tensorrt_onnxruntime"

    def infer(self, user_cat: torch.Tensor, user_num: torch.Tensor, user_multi: torch.Tensor) -> torch.Tensor:
        # ORT expects CPU numpy inputs.
        inp = {
            "user_cat": user_cat.detach().cpu().numpy().astype(np.int64, copy=False),
            "user_num": user_num.detach().cpu().numpy().astype(np.float32, copy=False),
            "user_multi": user_multi.detach().cpu().numpy().astype(np.int64, copy=False),
        }
        out = self.session.run([self.output_name], inp)[0]
        return torch.from_numpy(np.asarray(out, dtype=np.float32)).to(self.device, non_blocking=True)

