from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunLog:
    path: Path

    def write(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts} | {msg}\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def _default_logs_dir() -> Path:
    # Use cwd so notebooks + CLI runs land in the same place.
    return Path(os.getcwd()).resolve() / "logs"


def start_run_log(*, kind: str, name: str | None = None, logs_dir: str | Path | None = None) -> RunLog:
    d = Path(logs_dir).resolve() if logs_dir is not None else _default_logs_dir()
    safe = (name or "run").replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    p = d / f"{kind}_{safe}_{stamp}.log"
    rl = RunLog(path=p)
    rl.write(f"START kind={kind} name={name or ''} host={socket.gethostname()} pid={os.getpid()}")
    return rl

