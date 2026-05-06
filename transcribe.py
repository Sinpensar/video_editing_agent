"""
Transcribe a video file into timestamped segments using Whisper.

The result is cached to workspace/<sha1>.json so re-running on the same
video is essentially free.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import whisper

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

_MODEL_CACHE: dict[str, "whisper.Whisper"] = {}


def _file_fingerprint(path: Path) -> str:
    """Hash the first + last 1MB + size — fast and good enough to detect changes."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, os.SEEK_END)
            h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono PCM audio with ffmpeg (Whisper's preferred input)."""
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def _load_model(name: str) -> "whisper.Whisper":
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = whisper.load_model(name)
    return _MODEL_CACHE[name]


def transcribe(
    video_path: str | Path,
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Transcribe a video file. Returns:
        {
            "video": str,
            "language": str,
            "duration": float,           # seconds
            "segments": [
                {"id": int, "start": float, "end": float, "text": str},
                ...
            ],
        }

    Cached by file fingerprint + model name so repeat calls are instant.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_name = model_name or os.getenv("WHISPER_MODEL", "base")
    fp = _file_fingerprint(video_path)
    cache_path = WORKSPACE / f"{video_path.stem}.{fp}.{model_name}.json"

    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        _extract_audio(video_path, audio_path)

        model = _load_model(model_name)
        result = model.transcribe(str(audio_path), language=language)

    segments = [
        {
            "id": int(seg["id"]),
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    duration = segments[-1]["end"] if segments else 0.0

    output = {
        "video": str(video_path),
        "language": result.get("language", "unknown"),
        "duration": duration,
        "segments": segments,
    }

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_file>")
        sys.exit(1)
    out = transcribe(sys.argv[1])
    print(f"Language: {out['language']}, Duration: {format_time(out['duration'])}, "
          f"{len(out['segments'])} segments")
    for s in out["segments"][:10]:
        print(f"  [{format_time(s['start'])} - {format_time(s['end'])}] {s['text']}")
    if len(out["segments"]) > 10:
        print(f"  ... +{len(out['segments']) - 10} more")
