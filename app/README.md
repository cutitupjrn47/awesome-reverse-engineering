# Local voice-swap API

This is an opt-in, local FastAPI service for converting audio with RVC voice models. It binds to `127.0.0.1` when started as shown below; do not expose it to a network without adding authentication and upload controls.

## Requirements

- Python 3.10–3.12
- FFmpeg available on `PATH`
- A working PyTorch 2.5.x install; install matching `torch`, `torchaudio`, and `torchvision` CPU or CUDA builds before the remaining requirements
- An RVC `.pth` checkpoint for each voice; `.index` files are optional

Install packages from an isolated environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
.\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

Place each voice in a folder under `app\models`, for example:

```text
app/
  models/
    demo/
      voice.pth
      voice.index
      voice.json
```

`voice.json` is optional and may contain a display name:

```json
{ "name": "Demo voice" }
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`GET /voices` lists valid checkpoints. Submit multipart form data to `POST /swap` with an `audio` file and a `voice_id`. Optional fields are `pitch_shift` (-24 to 24), `index_rate` (0 to 1), `f0_method` (`rmvpe+`, `rmvpe`, `pm`, or `crepe`), `separate` (default true), `denoise` (default false), and `dereverb` (default false). The response contains a job ID. Poll `GET /jobs/{id}`, cancel queued/running jobs with `DELETE /jobs/{id}`, and download completed WAV output from `GET /jobs/{id}/result`.

The first separation can download Demucs weights, and the first conversion can download HuBERT and RMVPE assets through their libraries. Set `VOICE_HUBERT_PATH` and `VOICE_RMVPE_PATH` to local assets to avoid the RVC downloads. Only install RVC checkpoints you trust: the upstream loader uses PyTorch checkpoint deserialization. `VOICE_DEVICE` accepts `cpu` (default) or `cuda`; `VOICE_MODELS_DIR`, `VOICE_WORK_DIR`, upload size, duration, and queue limits can also be set through environment variables.

Demucs separates vocals from accompaniment before conversion by default. Denoising and WPE dereverberation are opt-in; both add processing time and may change vocal quality. RVC inference performs its own chunked feature/pitch/inference processing. The service hashes each uploaded source with SHA-256, caps uploads and decoded duration, limits concurrent work, and keeps jobs in memory until the service restarts.

Job files are stored under `VOICE_WORK_DIR`. Failed and cancelled jobs are removed when processing stops; successful jobs keep only the downloadable WAV result. On startup, leftover directories named for prior job UUIDs are removed, so completed results are available only until the next restart.

Run the app tests from the repository root with `python -m unittest tests.test_voice_app -v`.
```
```