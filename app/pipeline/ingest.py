from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioInputError(ValueError):
    pass


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(
    source_path: Path, decoded_path: Path, max_duration_seconds: int
) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to decode uploaded audio")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-t",
                str(max_duration_seconds + 1),
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(decoded_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioInputError("Audio decoding timed out") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip()[-500:] if error.stderr else "invalid audio"
        raise AudioInputError(f"Could not decode uploaded audio: {detail}") from error

    try:
        info = sf.info(decoded_path)
        if info.frames <= 0 or info.duration > max_duration_seconds:
            raise AudioInputError(
                f"Audio must be between 0 and {max_duration_seconds} seconds"
            )
        audio, sample_rate = sf.read(decoded_path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise AudioInputError("Decoded audio could not be read") from error
    if sample_rate != 44100 or audio.shape[1] != 2:
        raise AudioInputError("Decoded audio has an unsupported format")
    if not np.isfinite(audio).all():
        raise AudioInputError("Audio contains non-finite samples")
    return audio, sample_rate
