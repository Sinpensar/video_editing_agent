"""
Pre-flight validation for `create_cut` calls.

The LLM proposes a list of (video_id, start, end) clips. Before we hand
that to ffmpeg, we run sanity checks. If any fire, we return the list of
issues to the LLM as a tool result and let it self-correct.

Two layers:

  1. Sanity checks (always run): malformed clips that would either crash
     ffmpeg or produce silently-wrong output.
       - unknown video_id, negative start, end <= start
       - end beyond video duration
       - clip too short (<min_clip_duration s) or suspiciously long
       - exact duplicates, large overlaps within the same source video

  2. Goal checks (only if a goal is set via Project.set_goal):
       - total duration within ±tolerance of target
       - clip count within [min_clips, max_clips]
       - every must-include keyword shows up somewhere in the kept content
       - no must-exclude keyword shows up in the kept content
"""

from __future__ import annotations

from typing import Any, Iterable

# ---- defaults --------------------------------------------------------------

DEFAULT_MIN_CLIP_DURATION = 0.3       # seconds
DEFAULT_MAX_CLIP_DURATION = 90.0      # seconds — above this, very likely a mistake
DEFAULT_DURATION_OVERSHOOT = 0.5      # how far past the video's known end we tolerate
DEFAULT_OVERLAP_TOLERANCE = 0.5       # seconds of overlap before we complain
DEFAULT_TARGET_TOLERANCE = 0.25       # ±25% of the target duration


def _clipped_text(clips: list[dict], project: Any) -> str:
    """Concatenate every transcript segment that overlaps any clip window.

    Used by the goal-keyword checks. Lower-cased for case-insensitive matching.
    """
    parts: list[str] = []
    for c in clips:
        entry = project.videos.get(c.get("video_id"))
        if not entry:
            continue
        s, e = float(c.get("start", 0)), float(c.get("end", 0))
        for seg in entry.transcript.get("segments", []):
            if seg["end"] > s and seg["start"] < e:
                parts.append(seg["text"])
    return " ".join(parts).lower()


def validate_clips(
    clips: Iterable[dict],
    project: Any,
    goal: dict | None = None,
    *,
    min_clip_duration: float = DEFAULT_MIN_CLIP_DURATION,
    max_clip_duration: float = DEFAULT_MAX_CLIP_DURATION,
    overshoot_tolerance: float = DEFAULT_DURATION_OVERSHOOT,
    overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
    target_tolerance: float = DEFAULT_TARGET_TOLERANCE,
) -> list[str]:
    """Return a list of human-readable issue strings. Empty list = passes.

    `project` only needs to expose `.videos[video_id]` with a `.transcript`
    dict (so this function works in tests without a real Project class).
    """
    clips = list(clips)
    issues: list[str] = []

    if not clips:
        return ["No clips were provided."]

    total = 0.0

    # -- per-clip checks -----------------------------------------------------
    for i, c in enumerate(clips):
        vid = c.get("video_id")
        entry = project.videos.get(vid)
        if not entry:
            issues.append(
                f"clip[{i}] references unknown video_id={vid!r}. "
                f"Known: {list(project.videos)}"
            )
            continue

        try:
            start = float(c.get("start"))
            end = float(c.get("end"))
        except (TypeError, ValueError):
            issues.append(f"clip[{i}] has non-numeric start/end: {c}")
            continue

        if start < 0:
            issues.append(f"clip[{i}] start ({start:.2f}s) is negative")
        if end <= start:
            issues.append(
                f"clip[{i}] end ({end:.2f}s) must be greater than start "
                f"({start:.2f}s)"
            )
            continue

        v_dur = float(entry.transcript.get("duration") or 0)
        if v_dur and end > v_dur + overshoot_tolerance:
            issues.append(
                f"clip[{i}] end ({end:.2f}s) exceeds duration of {vid} "
                f"({v_dur:.2f}s)"
            )

        d = end - start
        if d < min_clip_duration:
            issues.append(
                f"clip[{i}] is too short ({d:.2f}s < {min_clip_duration}s)"
            )
        if d > max_clip_duration:
            issues.append(
                f"clip[{i}] is suspiciously long ({d:.1f}s > {max_clip_duration}s); "
                f"did you mean to split it into smaller pieces?"
            )

        total += max(0.0, d)

    # -- duplicates ----------------------------------------------------------
    seen_keys: set[tuple] = set()
    for i, c in enumerate(clips):
        try:
            key = (
                c.get("video_id"),
                round(float(c.get("start", 0)), 2),
                round(float(c.get("end", 0)), 2),
            )
        except (TypeError, ValueError):
            continue
        if key in seen_keys:
            issues.append(f"clip[{i}] duplicates an earlier clip")
        seen_keys.add(key)

    # -- overlaps within same video -----------------------------------------
    by_video: dict[str, list[tuple[int, float, float]]] = {}
    for i, c in enumerate(clips):
        vid = c.get("video_id")
        if not vid:
            continue
        try:
            s, e = float(c.get("start", 0)), float(c.get("end", 0))
        except (TypeError, ValueError):
            continue
        if e > s:
            by_video.setdefault(vid, []).append((i, s, e))

    for vid, items in by_video.items():
        items.sort(key=lambda x: x[1])
        for j in range(1, len(items)):
            pi, ps, pe = items[j - 1]
            ci, cs, ce = items[j]
            overlap = pe - cs
            if overlap > overlap_tolerance:
                issues.append(
                    f"clip[{ci}] overlaps clip[{pi}] in {vid} by "
                    f"{overlap:.2f}s"
                )

    # -- goal checks (only if a goal was registered) ------------------------
    goal = goal or {}

    target = goal.get("target_duration_sec")
    if target and float(target) > 0:
        target = float(target)
        delta = abs(total - target) / target
        if delta > target_tolerance:
            issues.append(
                f"total duration {total:.1f}s deviates from target "
                f"{target:.1f}s by {delta * 100:.0f}% "
                f"(tolerance {int(target_tolerance * 100)}%)"
            )

    min_clips = goal.get("min_clips")
    if isinstance(min_clips, int) and min_clips > 0 and len(clips) < min_clips:
        issues.append(
            f"only {len(clips)} clip(s) selected, target requires at least {min_clips}"
        )
    max_clips = goal.get("max_clips")
    if isinstance(max_clips, int) and max_clips > 0 and len(clips) > max_clips:
        issues.append(
            f"{len(clips)} clip(s) selected, target allows at most {max_clips}"
        )

    must_include = goal.get("must_include_keywords") or []
    must_exclude = goal.get("must_exclude_keywords") or []
    if must_include or must_exclude:
        text = _clipped_text(clips, project)
        for kw in must_include:
            if kw and kw.lower() not in text:
                issues.append(
                    f"required keyword {kw!r} does not appear anywhere in the kept content"
                )
        for kw in must_exclude:
            if kw and kw.lower() in text:
                issues.append(
                    f"forbidden keyword {kw!r} appears in the kept content"
                )

    return issues
