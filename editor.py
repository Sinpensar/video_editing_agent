"""
FFmpeg-based cutting and concatenation.

Given a list of clips like:
    [{"video": "/path/a.mp4", "start": 12.3, "end": 24.5}, ...]
produce a single output mp4 with all clips concatenated in order.

Strategy:
- Re-encode each clip individually (so cut points are frame-accurate).
- Concatenate them losslessly via the concat demuxer.

Re-encoding is slower but avoids the keyframe-alignment artifacts you get
with `-c copy` cuts. For an MVP that's the right trade-off.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, TypedDict

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class Clip(TypedDict):
    video: str
    start: float
    end: float


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  stderr: {result.stderr[-1500:]}"
        )


def _cut_one(clip: Clip, out_path: Path) -> None:
    duration = max(0.01, float(clip["end"]) - float(clip["start"]))
    cmd = [
        "ffmpeg",
        "-ss", f"{float(clip['start']):.3f}",
        "-i", str(clip["video"]),
        "-t", f"{duration:.3f}",
        # re-encode for accurate cuts and uniform codec across clips
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(out_path),
    ]
    _run(cmd)


def _concat(parts: list[Path], out_path: Path) -> None:
    # Use the concat demuxer with a list file. Lossless for matching codecs.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for p in parts:
            # ffmpeg concat list quoting
            f.write(f"file '{p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")
    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            "-y",
            str(out_path),
        ]
        _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


def assemble(clips: Iterable[Clip], output_filename: str | None = None) -> Path:
    """
    Cut each clip and concatenate them into a single mp4.
    Returns the absolute path of the resulting file.
    """
    clips = [c for c in clips if float(c["end"]) > float(c["start"])]
    if not clips:
        raise ValueError("No valid clips to assemble.")

    if not output_filename:
        output_filename = f"vlog_{uuid.uuid4().hex[:8]}.mp4"
    if not output_filename.lower().endswith(".mp4"):
        output_filename += ".mp4"
    out_path = OUTPUT_DIR / output_filename

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        for i, clip in enumerate(clips):
            part_path = tmp_dir / f"part_{i:03d}.mp4"
            _cut_one(clip, part_path)
            parts.append(part_path)

        if len(parts) == 1:
            # Just move/copy the single piece
            import shutil
            shutil.copy2(parts[0], out_path)
        else:
            _concat(parts, out_path)

    return out_path


def probe_duration(video_path: str | Path) -> float:
    """Return duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip() or 0.0)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python editor.py plan.json")
        print('plan.json: [{"video": "...", "start": 0, "end": 5}, ...]')
        sys.exit(1)
    with open(sys.argv[1]) as f:
        plan = json.load(f)
    out = assemble(plan)
    print(f"Wrote {out}")
