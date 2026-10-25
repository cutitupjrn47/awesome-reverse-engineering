from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def clean_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    denoise: bool = False,
    dereverb: bool = False,
) -> np.ndarray:
    cleaned = np.asarray(audio, dtype=np.float32).reshape(-1)
    if denoise:
        import noisereduce

        cleaned = noisereduce.reduce_noise(
            y=cleaned, sr=sample_rate, stationary=False, prop_decrease=0.65
        ).astype(np.float32, copy=False)
    if dereverb:
        from nara_wpe.wpe import wpe

        _, _, spectrum = stft(
            cleaned,
            fs=sample_rate,
            nperseg=512,
            noverlap=384,
            boundary="zeros",
            padded=True,
        )
        dereverberated = wpe(spectrum, taps=10, delay=3, iterations=3)
        _, cleaned = istft(
            dereverberated,
            fs=sample_rate,
            nperseg=512,
            noverlap=384,
            boundary=True,
        )
        cleaned = cleaned.astype(np.float32, copy=False)
    if not np.isfinite(cleaned).all():
        raise RuntimeError("Audio cleanup produced non-finite samples")
    return cleaned
