from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path


VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
logger = logging.getLogger(__name__)


class VoiceNotFoundError(LookupError):
    pass


class InvalidVoiceError(ValueError):
    pass


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    model_path: Path
    index_path: Path | None


class VoiceRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.resolve()

    def _safe_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.model_dir)
            return True
        except ValueError:
            return False

    def list(self) -> list[Voice]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        voices: list[Voice] = []
        for directory in sorted(self.model_dir.iterdir()):
            if not directory.is_dir() or not VOICE_ID.fullmatch(directory.name):
                continue
            try:
                voices.append(self.get(directory.name))
            except InvalidVoiceError as error:
                logger.warning("Skipping invalid voice model %s: %s", directory, error)
            except VoiceNotFoundError:
                continue
        return voices

    def get(self, voice_id: str) -> Voice:
        if not VOICE_ID.fullmatch(voice_id):
            raise VoiceNotFoundError(voice_id)
        directory = self.model_dir / voice_id
        if not directory.is_dir() or not self._safe_path(directory):
            raise VoiceNotFoundError(voice_id)

        checkpoints = sorted(directory.glob("*.pth"))
        indexes = sorted(directory.glob("*.index"))
        if len(checkpoints) != 1 or len(indexes) > 1:
            raise InvalidVoiceError(
                f"Voice '{voice_id}' must contain exactly one .pth file and at most one .index file"
            )
        checkpoint = checkpoints[0]
        index = indexes[0] if indexes else None
        if not self._safe_path(checkpoint) or (index and not self._safe_path(index)):
            raise InvalidVoiceError(f"Voice '{voice_id}' contains an unsafe file path")

        display_name = voice_id
        metadata = directory / "voice.json"
        if metadata.exists() or metadata.is_symlink():
            if not self._safe_path(metadata) or not metadata.is_file():
                raise InvalidVoiceError(
                    f"Voice '{voice_id}' contains an invalid voice.json"
                )
            try:
                contents = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise InvalidVoiceError(f"Invalid voice.json for '{voice_id}'") from error
            if not isinstance(contents, dict):
                raise InvalidVoiceError(f"voice.json for '{voice_id}' must be an object")
            name = contents.get("name", voice_id)
            if not isinstance(name, str) or not name.strip() or len(name) > 100:
                raise InvalidVoiceError(
                    f"voice.json for '{voice_id}' has an invalid name"
                )
            display_name = name.strip()

        return Voice(
            voice_id,
            display_name,
            checkpoint.resolve(),
            index.resolve() if index else None,
        )
