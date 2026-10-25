from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    divisor = gcd(source_rate, target_rate)
    return resample_poly(
        audio, target_rate // divisor, source_rate // divisor, axis=0
    ).astype(np.float32, copy=False)


def mix_audio(
    reference_vocals: np.ndarray,
    converted_vocals: np.ndarray,
    converted_rate: int,
    accompaniment: np.ndarray,
    output_rate: int,
) -> np.ndarray:
    reference = np.asarray(reference_vocals, dtype=np.float32)
    if reference.ndim == 2:
        reference = reference.mean(axis=1)
    converted = _resample(
        np.asarray(converted_vocals, dtype=np.float32).reshape(-1),
        converted_rate,
        output_rate,
    )
    backing = np.asarray(accompaniment, dtype=np.float32)
    if backing.ndim == 1:
        backing = backing[:, None]
    if backing.shape[1] == 1:
        backing = np.repeat(backing, 2, axis=1)

    target_length = len(backing)
    reference = _resample(reference, output_rate, output_rate)
    if len(reference) != target_length:
        reference = np.pad(
            reference[:target_length],
            (0, max(0, target_length - len(reference))),
        )
    converted = np.pad(
        converted[:target_length], (0, max(0, target_length - len(converted)))
    )

    reference_rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
    converted_rms = float(np.sqrt(np.mean(np.square(converted), dtype=np.float64)))
    if converted_rms > 1e-8 and reference_rms > 1e-8:
        converted *= min(reference_rms / converted_rms, 4.0)
    vocal_stereo = np.repeat(converted[:, None], 2, axis=1)
    mixed = backing + vocal_stereo
    peak = float(np.max(np.abs(mixed), initial=0.0))
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed.astype(np.float32, copy=False)


def export_audio(audio: np.ndarray, output_path: Path, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, format="WAV", subtype="PCM_16")
