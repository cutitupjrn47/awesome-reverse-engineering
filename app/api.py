from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.config import Settings
from app.jobs import (
    JobConflictError,
    JobManager,
    JobNotFoundError,
    QueueFullError,
)
from app.voices import InvalidVoiceError, VoiceNotFoundError, VoiceRegistry
from app.pipeline.convert import ConversionOptions
from app.pipeline.ingest import hash_file
from app.pipeline.run import SwapOptions, run_pipeline


logger = logging.getLogger(__name__)
settings = Settings()
settings.validate()
voices = VoiceRegistry(settings.model_dir)
jobs = JobManager(settings.max_workers, settings.max_active_jobs)
app = FastAPI(title="Local Voice Swap", version="1.0.0")

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}


@app.get("/voices")
def list_voices() -> list[dict[str, object]]:
    try:
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "has_index": voice.index_path is not None,
            }
            for voice in voices.list()
        ]
    except OSError as error:
        logger.exception("Could not list voice models")
        raise HTTPException(500, "Could not read voice models") from error


@app.post("/swap", status_code=status.HTTP_202_ACCEPTED)
async def create_swap(
    audio: UploadFile = File(...),
    voice_id: str = Form(...),
    pitch_shift: int = Form(0, ge=-24, le=24),
    index_rate: float = Form(0.5, ge=0.0, le=1.0),
    f0_method: str = Form("rmvpe+"),
    separate: bool = Form(True),
    denoise: bool = Form(False),
    dereverb: bool = Form(False),
) -> dict[str, object]:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported audio format")
    if f0_method not in {"rmvpe+", "rmvpe", "pm", "crepe"}:
        raise HTTPException(422, "Unsupported pitch extraction method")
    try:
        voice = voices.get(voice_id)
    except VoiceNotFoundError as error:
        raise HTTPException(404, "Voice model not found") from error
    except InvalidVoiceError as error:
        raise HTTPException(422, str(error)) from error

    job_id = str(uuid4())
    work_dir = settings.work_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=False)
    input_path = work_dir / f"input{suffix}"
    size = 0
    try:
        with input_path.open("wb") as destination:
            while block := await audio.read(1024 * 1024):
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "Uploaded audio exceeds the size limit")
                destination.write(block)
        if size == 0:
            raise HTTPException(400, "Uploaded audio is empty")
        input_hash = hash_file(input_path)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        await audio.close()

    options = SwapOptions(
        separate=separate,
        denoise=denoise,
        dereverb=dereverb,
        conversion=ConversionOptions(
            pitch_shift=pitch_shift, index_rate=index_rate, f0_method=f0_method
        ),
    )

    def task(context):
        return run_pipeline(
            context,
            input_path,
            work_dir / "converted.wav",
            voice,
            options,
            settings,
        )

    try:
        job = jobs.submit(job_id, work_dir, input_hash, task)
    except QueueFullError as error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(429, str(error)) from error
    return job.public()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        return jobs.get(job_id).public()
    except JobNotFoundError as error:
        raise HTTPException(404, "Job not found") from error


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, object]:
    try:
        return jobs.cancel(job_id).public()
    except JobNotFoundError as error:
        raise HTTPException(404, "Job not found") from error
    except JobConflictError as error:
        raise HTTPException(409, str(error)) from error


@app.get("/jobs/{job_id}/result")
def download_result(job_id: str) -> FileResponse:
    try:
        job = jobs.get(job_id)
    except JobNotFoundError as error:
        raise HTTPException(404, "Job not found") from error
    if job.status != "succeeded" or job.result_path is None:
        raise HTTPException(409, "Job has no completed result")
    if not job.result_path.is_file():
        raise HTTPException(410, "Completed result is no longer available")
    return FileResponse(
        job.result_path,
        media_type="audio/wav",
        filename="converted.wav",
    )
