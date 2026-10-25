from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    model_dir: Path = _path_from_env("VOICE_MODELS_DIR", ROOT_DIR / "app" / "models")
    work_dir: Path = _path_from_env(
        "VOICE_WORK_DIR", Path(tempfile.gettempdir()) / "voice-swap-jobs"
    )
    device: str = os.environ.get("VOICE_DEVICE", "cpu").lower()
    max_upload_bytes: int = int(os.environ.get("VOICE_MAX_UPLOAD_BYTES", 200_000_000))
    max_duration_seconds: int = int(
        os.environ.get("VOICE_MAX_DURATION_SECONDS", 900)
    )
    max_workers: int = int(os.environ.get("VOICE_MAX_WORKERS", 1))
    max_active_jobs: int = int(os.environ.get("VOICE_MAX_ACTIVE_JOBS", 8))
    hubert_path: str | None = os.environ.get("VOICE_HUBERT_PATH")
    rmvpe_path: str | None = os.environ.get("VOICE_RMVPE_PATH")

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("VOICE_DEVICE must be 'cpu' or 'cuda'")
        if self.max_upload_bytes < 1:
            raise ValueError("VOICE_MAX_UPLOAD_BYTES must be positive")
        if self.max_duration_seconds < 1:
            raise ValueError("VOICE_MAX_DURATION_SECONDS must be positive")
        if self.max_workers < 1 or self.max_active_jobs < self.max_workers:
            raise ValueError("VOICE_MAX_ACTIVE_JOBS must be >= VOICE_MAX_WORKERS >= 1")
