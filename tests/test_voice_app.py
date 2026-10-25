from __future__ import annotations

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
