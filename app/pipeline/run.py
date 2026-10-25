from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import Settings
from app.jobs import JobContext
from app.voices import Voice
from app.pipeline.clean import clean_audio
from app.pipeline.convert import ConversionOptions, convert_audio
from app.pipeline.ingest import decode_audio
from app.pipeline.mix import export_audio, mix_audio
from app.pipeline.separate import separate_vocals


@dataclass(frozen=True)
class SwapOptions:
    separate: bool = True
    denoise: bool = False
    dereverb: bool = False
    conversion: ConversionOptions = ConversionOptions()


def run_pipeline(
    context: JobContext,
    input_path: Path,
    output_path: Path,
    voice: Voice,
    options: SwapOptions,
    settings: Settings,
) -> Path:
    context.check_cancelled()
    context.update(5, "decoding")
    decoded_path = context.work_dir / "decoded.wav"
    source, sample_rate = decode_audio(
        input_path,
        decoded_path,
        settings.max_duration_seconds,
        cancel_event=context.cancel_event,
    )
    context.check_cancelled()

    if options.separate:
        context.update(20, "separating")
        vocals_path, accompaniment_path = separate_vocals(
            decoded_path, context.work_dir / "separated", settings.device,
            context.cancel_event,
        )
        vocals, vocal_rate = sf.read(
            vocals_path, dtype="float32", always_2d=True
        )
        accompaniment, backing_rate = sf.read(
            accompaniment_path, dtype="float32", always_2d=True
        )
        if vocal_rate != sample_rate or backing_rate != sample_rate:
            raise RuntimeError("Demucs returned stems at an unexpected sample rate")
    else:
        vocals = source
        accompaniment = np.zeros_like(source)
    context.check_cancelled()

    context.update(45, "cleaning")
    mono_vocals = vocals.mean(axis=1)
    cleaned = clean_audio(
        mono_vocals,
        sample_rate,
        denoise=options.denoise,
        dereverb=options.dereverb,
    )
    context.check_cancelled()

    context.update(60, "converting")
    converted, converted_rate = convert_audio(
        cleaned, sample_rate, voice, options.conversion, settings
    )
    context.check_cancelled()

    context.update(90, "mixing")
    mixed = mix_audio(
        mono_vocals, converted, converted_rate, accompaniment, sample_rate
    )
    export_audio(mixed, output_path, sample_rate)
    context.check_cancelled()
    return output_path
