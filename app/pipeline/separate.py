from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Event

from app.jobs import JobCancelledError


def separate_vocals(
    input_path: Path,
    output_dir: Path,
    device: str,
    cancel_event: Event,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "--name",
        "htdemucs",
        "--device",
        device,
        "--out",
        str(output_dir),
        str(input_path),
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_log:
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=error_log, text=True
        )
        while process.poll() is None:
            if cancel_event.wait(0.25):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise JobCancelledError("Job was cancelled during source separation")
        if process.returncode:
            error_log.seek(0, 2)
            error_log.seek(max(0, error_log.tell() - 1000))
            detail = error_log.read().strip() or f"Demucs exited with {process.returncode}"
            raise RuntimeError(f"Source separation failed: {detail}")

    stem_dir = output_dir / "htdemucs" / input_path.stem
    vocals = stem_dir / "vocals.wav"
    accompaniment = stem_dir / "no_vocals.wav"
    if not vocals.is_file() or not accompaniment.is_file():
        raise RuntimeError("Demucs completed without producing both expected stems")
    return vocals, accompaniment
