import json
import os
import shutil
import uuid
from pathlib import Path


class VoiceRegistry:
    def __init__(self, runtime_dir: str):
        self.runtime_dir = Path(runtime_dir)
        self.voice_dir = self.runtime_dir / "voices"
        self.registry_path = self.runtime_dir / "voice_registry.json"
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _save_registry(self):
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(self.registry, handle, ensure_ascii=False, indent=2, sort_keys=True)

    def list_voices(self) -> list[dict]:
        items = []
        for voice_id, metadata in sorted(self.registry.items()):
            item = dict(metadata)
            item["voice_id"] = voice_id
            items.append(item)
        return items

    def get_voice(self, voice_id: str) -> dict | None:
        voice = self.registry.get(voice_id)
        if voice is None:
            return None
        item = dict(voice)
        item["voice_id"] = voice_id
        return item

    def register_voice(
        self,
        source_path: str,
        filename: str,
        prompt_text: str,
        prompt_lang: str,
        name: str | None = None,
    ) -> dict:
        voice_id = uuid.uuid4().hex
        suffix = Path(filename or source_path).suffix.lower() or ".wav"
        dest_dir = self.voice_dir / voice_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"reference{suffix}"
        shutil.copy2(source_path, dest_path)
        metadata = {
            "name": name or voice_id,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "ref_audio_path": str(dest_path),
            "filename": os.path.basename(filename or dest_path.name),
        }
        self.registry[voice_id] = metadata
        self._save_registry()
        return self.get_voice(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        metadata = self.registry.pop(voice_id, None)
        if metadata is None:
            return False
        voice_dir = self.voice_dir / voice_id
        shutil.rmtree(voice_dir, ignore_errors=True)
        self._save_registry()
        return True
