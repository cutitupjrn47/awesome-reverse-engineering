from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app import api as api_module
from app.config import Settings
from app.jobs import JobManager, QueueFullError
from app.voices import InvalidVoiceError, Voice, VoiceNotFoundError, VoiceRegistry
from app.pipeline.clean import clean_audio
from app.pipeline.convert import ConversionOptions, RVCConverter
from app.pipeline.ingest import AudioInputError, decode_audio
from app.pipeline.mix import mix_audio
from app.pipeline.run import SwapOptions, run_pipeline


class VoiceRegistryTests(unittest.TestCase):
    def test_lists_voice_with_optional_index_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voice_dir = root / "demo"
            voice_dir.mkdir()
            (voice_dir / "model.pth").write_bytes(b"checkpoint")
            (voice_dir / "features.index").write_bytes(b"index")
            (voice_dir / "voice.json").write_text('{"name":"Demo"}', encoding="utf-8")

            voice = VoiceRegistry(root).get("demo")

            self.assertEqual(voice.name, "Demo")
            self.assertTrue(voice.model_path.is_file())
            self.assertEqual(voice.index_path.name, "features.index")

    def test_rejects_ambiguous_checkpoint_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voice_dir = root / "demo"
            voice_dir.mkdir()
            (voice_dir / "first.pth").touch()
            (voice_dir / "second.pth").touch()

            with self.assertRaises(InvalidVoiceError):
                VoiceRegistry(root).get("demo")

    def test_rejects_invalid_voice_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(VoiceNotFoundError):
                VoiceRegistry(Path(temporary)).get("../outside")


class JobManagerTests(unittest.TestCase):
    def test_records_success_and_progress(self):
        manager = JobManager(max_workers=1, max_active_jobs=1)
        job_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)

            def run(context):
                output = work_dir / "result.wav"
                context.update(50, "converting")
                output.write_bytes(b"wav")
                return output

            job = manager.submit(job_id, work_dir, "hash", run)
            completed = manager.get(job_id)
            for _ in range(100):
                if completed.status in {"succeeded", "failed"}:
                    break
                threading.Event().wait(0.01)
                completed = manager.get(job_id)

            self.assertEqual(job.id, job_id)
            self.assertEqual(completed.status, "succeeded")
            self.assertEqual(completed.progress, 100)
            self.assertTrue(completed.result_path.is_file())
            for _ in range(100):
                if list(work_dir.iterdir()) == [completed.result_path]:
                    break
                threading.Event().wait(0.01)
            self.assertEqual(list(work_dir.iterdir()), [completed.result_path])

    def test_rejects_when_active_job_limit_is_reached(self):
        manager = JobManager(max_workers=1, max_active_jobs=1)
        started = threading.Event()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)

            def wait_for_release(context):
                started.set()
                release.wait(2)
                context.check_cancelled()
                result = work_dir / "result.wav"
                result.touch()
                return result

            first_id = str(uuid4())
            manager.submit(first_id, work_dir, "hash", wait_for_release)
            self.assertTrue(started.wait(1))
            with self.assertRaises(QueueFullError):
                manager.submit(str(uuid4()), work_dir, "hash", wait_for_release)
            manager.cancel(first_id)
            release.set()
            for _ in range(100):
                if manager.get(first_id).status == "cancelled":
                    break
                threading.Event().wait(0.01)
            self.assertEqual(manager.get(first_id).status, "cancelled")
            self.assertFalse(work_dir.exists())

    def test_cancelling_queued_jobs_cleans_uploads_and_frees_capacity(self):
        manager = JobManager(max_workers=1, max_active_jobs=2)
        started = threading.Event()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            running_dir = root / str(uuid4())
            running_dir.mkdir()
            running_upload = running_dir / "input.wav"
            running_upload.write_bytes(b"running")

            def running_task(context):
                started.set()
                release.wait(2)
                context.check_cancelled()
                result = running_dir / "result.wav"
                result.touch()
                return result

            running_id = str(uuid4())
            manager.submit(running_id, running_dir, "hash", running_task)
            self.assertTrue(started.wait(1))

            queued_dir = root / str(uuid4())
            queued_dir.mkdir()
            (queued_dir / "input.wav").write_bytes(b"queued")
            queued_id = str(uuid4())
            manager.submit(queued_id, queued_dir, "hash", running_task)
            manager.cancel(queued_id)
            self.assertFalse(queued_dir.exists())

            replacement_dir = root / str(uuid4())
            replacement_dir.mkdir()
            replacement_id = str(uuid4())
            manager.submit(replacement_id, replacement_dir, "hash", running_task)
            manager.cancel(replacement_id)
            self.assertFalse(replacement_dir.exists())

            manager.cancel(running_id)
            self.assertTrue(running_upload.exists())
            release.set()
            for _ in range(100):
                if manager.get(running_id).status == "cancelled" and not running_dir.exists():
                    break
                threading.Event().wait(0.01)
            self.assertEqual(manager.get(running_id).status, "cancelled")
            self.assertFalse(running_dir.exists())

    def test_running_cancellation_wins_before_success_is_published(self):
        manager = JobManager(max_workers=1, max_active_jobs=1)
        validating_result = threading.Event()
        continue_validation = threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary) / str(uuid4())
            work_dir.mkdir()
            result_path = work_dir / "result.wav"

            class PausedResultPath:
                def is_file(self):
                    validating_result.set()
                    continue_validation.wait(2)
                    return result_path.is_file()

                def resolve(self):
                    return result_path.resolve()

            def complete(context):
                result_path.write_bytes(b"wav")
                return PausedResultPath()

            job_id = str(uuid4())
            manager.submit(job_id, work_dir, "hash", complete)
            self.assertTrue(validating_result.wait(1))
            manager.cancel(job_id)
            self.assertTrue(result_path.is_file())
            continue_validation.set()

            for _ in range(100):
                if manager.get(job_id).status == "cancelled" and not work_dir.exists():
                    break
                threading.Event().wait(0.01)
            self.assertEqual(manager.get(job_id).status, "cancelled")
            self.assertFalse(work_dir.exists())

    def test_failed_jobs_remove_their_work_directory(self):
        manager = JobManager(max_workers=1, max_active_jobs=1)
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary) / str(uuid4())
            work_dir.mkdir()

            def fail(context):
                (work_dir / "intermediate.wav").write_bytes(b"intermediate")
                raise RuntimeError("conversion failed")

            job_id = str(uuid4())
            manager.submit(job_id, work_dir, "hash", fail)
            for _ in range(100):
                if manager.get(job_id).status == "failed" and not work_dir.exists():
                    break
                threading.Event().wait(0.01)
            self.assertEqual(manager.get(job_id).status, "failed")
            self.assertFalse(work_dir.exists())

    def test_startup_cleanup_removes_only_stale_job_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale_dir = root / str(uuid4())
            stale_dir.mkdir()
            (stale_dir / "input.wav").touch()
            unrelated_dir = root / "keep-this"
            unrelated_dir.mkdir()

            settings = replace(api_module.settings, work_dir=root)
            with patch.object(api_module, "settings", settings):
                with TestClient(api_module.app):
                    pass

            self.assertFalse(stale_dir.exists())
            self.assertTrue(unrelated_dir.is_dir())


class MixTests(unittest.TestCase):
    def test_matches_vocal_loudness_and_returns_stereo_at_backing_length(self):
        reference = np.full(100, 0.2, dtype=np.float32)
        converted = np.full(50, 0.1, dtype=np.float32)
        backing = np.zeros((100, 2), dtype=np.float32)

        mixed = mix_audio(reference, converted, 22050, backing, 44100)

        self.assertEqual(mixed.shape, (100, 2))
        self.assertTrue(np.allclose(mixed[:, 0], mixed[:, 1]))
        self.assertAlmostEqual(float(np.sqrt(np.mean(mixed[:, 0] ** 2))), 0.2, places=3)


class CleanTests(unittest.TestCase):
    def test_cleanup_preserves_finite_audio_with_dereverberation(self):
        audio = np.sin(np.linspace(0, 20, 4096, dtype=np.float32))

        cleaned = clean_audio(audio, 16000, dereverb=True)

        self.assertEqual(cleaned.shape, audio.shape)
        self.assertTrue(np.isfinite(cleaned).all())


class ConverterTests(unittest.TestCase):
    def test_configures_voice_and_normalizes_integer_rvc_output(self):
        class FakeLoader:
            def __init__(self, **kwargs):
                self.options = None

            def apply_conf(self, **kwargs):
                self.options = kwargs

            def generate_from_cache(self, audio_data, tag):
                self.input = (audio_data, tag)
                return np.array([32767, -32768], dtype=np.int16), 22050

        fake_module = types.ModuleType("infer_rvc_python")
        fake_module.BaseLoader = FakeLoader
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "voice.pth"
            model_path.touch()
            voice = Voice("demo", "Demo", model_path, None)
            converter = RVCConverter(Settings(device="cpu"))
            with patch.dict("sys.modules", {"infer_rvc_python": fake_module}):
                audio, sample_rate = converter.convert(
                    np.zeros(100, dtype=np.float32),
                    16000,
                    voice,
                    ConversionOptions(),
                )

            self.assertEqual(sample_rate, 22050)
            self.assertTrue(np.allclose(audio, [32767 / 32768, -1.0]))
            self.assertEqual(converter._loader.options["tag"], "voice-demo")
            self.assertEqual(converter._loader.options["file_model"], str(model_path))


class IngestTests(unittest.TestCase):
    def test_limits_ffmpeg_output_and_rejects_audio_over_duration(self):
        command = []

        def run_ffmpeg(args, **kwargs):
            command.extend(args)

        with (
            patch("app.pipeline.ingest.shutil.which", return_value="ffmpeg"),
            patch("app.pipeline.ingest.subprocess.run", side_effect=run_ffmpeg),
            patch(
                "app.pipeline.ingest.sf.info",
                return_value=SimpleNamespace(frames=88200, duration=2),
            ),
            patch("app.pipeline.ingest.sf.read") as read_audio,
        ):
            with self.assertRaisesRegex(AudioInputError, "between 0 and 1 seconds"):
                decode_audio(Path("input.mp3"), Path("decoded.wav"), 1)

        self.assertEqual(command[command.index("-t") + 1], "2")
        self.assertFalse(read_audio.called)


class PipelineTests(unittest.TestCase):
    def test_pipeline_exports_converted_audio_when_separation_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "job"
            work_dir.mkdir()
            input_audio = np.full((441, 2), 0.1, dtype=np.float32)
            voice = Voice("demo", "Demo", root / "voice.pth", None)
            context = SimpleNamespace(
                work_dir=work_dir,
                cancel_event=threading.Event(),
                check_cancelled=lambda: None,
                update=lambda progress, stage: None,
            )
            with (
                patch(
                    "app.pipeline.run.decode_audio",
                    return_value=(input_audio, 44100),
                ),
                patch(
                    "app.pipeline.run.convert_audio",
                    return_value=(np.full(441, 0.05, dtype=np.float32), 44100),
                ),
            ):
                output = run_pipeline(
                    context,
                    root / "input.wav",
                    root / "converted.wav",
                    voice,
                    SwapOptions(separate=False),
                    Settings(),
                )

            converted, sample_rate = sf.read(output, always_2d=True)
            self.assertEqual(sample_rate, 44100)
            self.assertEqual(converted.shape, input_audio.shape)
            self.assertTrue(np.isfinite(converted).all())


class ApiTests(unittest.TestCase):
    def test_swap_rejects_oversized_request_before_multipart_parsing(self):
        max_upload_bytes = 1024
        settings = replace(api_module.settings, max_upload_bytes=max_upload_bytes)
        boundary = "upload-boundary"
        audio = b"x" * (
            max_upload_bytes + 2 * api_module.MULTIPART_OVERHEAD_BYTES
        )
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="input.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("ascii") + audio + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="voice_id"\r\n\r\n'
            "demo\r\n"
            f"--{boundary}--\r\n"
        ).encode("ascii")

        async def invoke(headers):
            chunk_size = 64 * 1024
            chunks = [
                body[offset : offset + chunk_size]
                for offset in range(0, len(body), chunk_size)
            ]
            received = 0
            response = {}

            async def receive():
                nonlocal received
                received += 1
                if received <= len(chunks):
                    return {
                        "type": "http.request",
                        "body": chunks[received - 1],
                        "more_body": received < len(chunks),
                    }
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                if message["type"] == "http.response.start":
                    response["status"] = message["status"]

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/swap",
                "raw_path": b"/swap",
                "query_string": b"",
                "headers": headers,
                "client": ("test", 123),
                "server": ("test", 80),
                "root_path": "",
                "state": {},
            }
            with patch.object(api_module, "settings", settings):
                await api_module.app(scope, receive, send)
            return response.get("status"), received, len(chunks)

        content_length_status, receive_calls, _ = asyncio.run(
            invoke(
                [
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode(),
                    ),
                    (b"content-length", str(len(body)).encode()),
                ]
            )
        )
        self.assertEqual(content_length_status, 413)
        self.assertEqual(receive_calls, 0)

        chunked_status, receive_calls, chunk_count = asyncio.run(
            invoke(
                [
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode(),
                    )
                ]
            )
        )
        self.assertEqual(chunked_status, 413)
        self.assertLess(receive_calls, chunk_count)

        invalid_length_status, receive_calls, chunk_count = asyncio.run(
            invoke(
                [
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode(),
                    ),
                    (b"content-length", b"not-a-length"),
                ]
            )
        )
        self.assertEqual(invalid_length_status, 413)
        self.assertLess(receive_calls, chunk_count)

    def test_swap_upload_creates_pollable_job_and_downloadable_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models"
            voice_dir = model_dir / "demo"
            voice_dir.mkdir(parents=True)
            (voice_dir / "voice.pth").touch()
            settings = replace(
                api_module.settings,
                model_dir=model_dir,
                work_dir=root / "jobs",
                max_upload_bytes=len(b"audio"),
                max_active_jobs=2,
                max_workers=1,
            )
            manager = JobManager(max_workers=1, max_active_jobs=2)

            def fake_pipeline(context, input_path, output_path, voice, options, config):
                self.assertEqual(input_path.read_bytes(), b"audio")
                self.assertEqual(voice.id, "demo")
                output_path.write_bytes(b"wav-result")
                return output_path

            with (
                patch.object(api_module, "settings", settings),
                patch.object(api_module, "voices", VoiceRegistry(model_dir)),
                patch.object(api_module, "jobs", manager),
                patch.object(api_module, "run_pipeline", fake_pipeline),
            ):
                client = TestClient(api_module.app)
                response = client.post(
                    "/swap",
                    files={"audio": ("input.wav", b"audio", "audio/wav")},
                    data={"voice_id": "demo", "separate": "false"},
                )
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["id"]
                for _ in range(100):
                    job = client.get(f"/jobs/{job_id}").json()
                    if job["status"] in {"succeeded", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(job["status"], "succeeded", job)
                result = client.get(f"/jobs/{job_id}/result")
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.content, b"wav-result")


if __name__ == "__main__":
    unittest.main()
