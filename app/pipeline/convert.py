from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.voices import Voice


@dataclass(frozen=True)
class ConversionOptions:
    pitch_shift: int = 0
    index_rate: float = 0.5
    f0_method: str = "rmvpe+"


class RVCConverter:
    """Lazily loads the RVC runtime and keeps its model cache across jobs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._loader = None

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice: Voice,
        options: ConversionOptions,
    ) -> tuple[np.ndarray, int]:
        if options.f0_method not in {"rmvpe+", "rmvpe", "pm", "crepe"}:
            raise ValueError("Unsupported pitch extraction method")
        if not -24 <= options.pitch_shift <= 24:
            raise ValueError("Pitch shift must be between -24 and 24 semitones")
        if not 0.0 <= options.index_rate <= 1.0:
            raise ValueError("Index rate must be between 0 and 1")

        with self._lock:
            if self._loader is None:
                for label, configured_path in (
                    ("HuBERT", self.settings.hubert_path),
                    ("RMVPE", self.settings.rmvpe_path),
                ):
                    if configured_path and not Path(configured_path).exists():
                        raise FileNotFoundError(
                            f"Configured {label} model was not found: {configured_path}"
                        )
                try:
                    # infer_rvc_python currently depends on librosa 0.9.x, which
                    # still references NumPy aliases removed in NumPy 1.24.
                    for alias, value in (
                        ("complex", complex),
                        ("float", float),
                        ("int", int),
                    ):
                        if alias not in np.__dict__:
                            setattr(np, alias, value)
                    from infer_rvc_python import BaseLoader
                except ImportError as error:
                    raise RuntimeError(
                        "RVC runtime is missing; install app/requirements.txt"
                    ) from error
                self._loader = BaseLoader(
                    only_cpu=self.settings.device == "cpu",
                    hubert_path=self.settings.hubert_path,
                    rmvpe_path=self.settings.rmvpe_path,
                )

            tag = f"voice-{voice.id}"
            self._loader.apply_conf(
                tag=tag,
                file_model=str(voice.model_path),
                pitch_algo=options.f0_method,
                pitch_lvl=options.pitch_shift,
                file_index=str(voice.index_path) if voice.index_path else "",
                index_influence=options.index_rate if voice.index_path else 0.0,
                respiration_median_filtering=3,
                envelope_ratio=0.25,
                consonant_breath_protection=0.33,
            )
            converted, output_rate = self._loader.generate_from_cache(
                audio_data=(np.asarray(audio, dtype=np.float32), sample_rate),
                tag=tag,
            )

        result = np.asarray(converted, dtype=np.float32)
        if result.size == 0 or not np.isfinite(result).all():
            raise RuntimeError("RVC inference returned invalid audio")
        if np.max(np.abs(result)) > 1.0:
            result /= 32768.0
        return result.reshape(-1), int(output_rate)


_converter: RVCConverter | None = None
_converter_lock = threading.Lock()


def convert_audio(
    audio: np.ndarray,
    sample_rate: int,
    voice: Voice,
    options: ConversionOptions,
    settings: Settings,
) -> tuple[np.ndarray, int]:
    global _converter
    with _converter_lock:
        if _converter is None or _converter.settings != settings:
            _converter = RVCConverter(settings)
        converter = _converter
    return converter.convert(audio, sample_rate, voice, options)
