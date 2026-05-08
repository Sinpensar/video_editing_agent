"""Unit tests for validators.py — fully covers the sanity checks and
goal checks against a synthetic project."""

from types import SimpleNamespace

import pytest

import validators


def _video(vid: str, dur: float, segments: list[dict]):
    return SimpleNamespace(
        video_id=vid,
        path=SimpleNamespace(name=f"{vid}.mp4"),
        transcript={"duration": dur, "segments": segments},
    )


@pytest.fixture
def project():
    return SimpleNamespace(videos={
        "v1": _video("v1", 60.0, [
            {"id": 0, "start": 0.0,  "end": 5.0,  "text": "intro to coffee"},
            {"id": 1, "start": 5.0,  "end": 12.0, "text": "grind beans fine"},
            {"id": 2, "start": 12.0, "end": 25.0, "text": "filler um er filler"},
            {"id": 3, "start": 25.0, "end": 40.0, "text": "pull espresso"},
            {"id": 4, "start": 40.0, "end": 60.0, "text": "talk about weather"},
        ]),
    })


# ---- sanity checks ---------------------------------------------------------

def test_clean_cut_passes(project):
    issues = validators.validate_clips([
        {"video_id": "v1", "start": 0.3, "end": 4.5},
        {"video_id": "v1", "start": 5.2, "end": 11.5},
    ], project)
    assert issues == []


def test_unknown_video_id(project):
    issues = validators.validate_clips(
        [{"video_id": "v9", "start": 0, "end": 5}], project)
    assert any("unknown video_id" in i for i in issues)


def test_negative_start(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": -1, "end": 5}], project)
    assert any("negative" in i for i in issues)


def test_end_not_greater_than_start(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 5, "end": 5}], project)
    assert any("greater than start" in i for i in issues)


def test_overshoot(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 50, "end": 65}], project)
    assert any("exceeds duration" in i for i in issues)


def test_too_short(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 0.1}], project)
    assert any("too short" in i for i in issues)


def test_suspiciously_long(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 200}], project)
    assert any("suspiciously long" in i for i in issues)


def test_duplicate_clips(project):
    issues = validators.validate_clips([
        {"video_id": "v1", "start": 0.5, "end": 5.0},
        {"video_id": "v1", "start": 0.5, "end": 5.0},
    ], project)
    assert any("duplicates" in i for i in issues)


def test_overlap_within_same_video(project):
    issues = validators.validate_clips([
        {"video_id": "v1", "start": 0, "end": 10},
        {"video_id": "v1", "start": 5, "end": 15},
    ], project)
    assert any("overlaps" in i for i in issues)


def test_empty_clip_list(project):
    issues = validators.validate_clips([], project)
    assert any("No clips" in i for i in issues)


def test_non_numeric_start(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": "abc", "end": 5}], project)
    assert any("non-numeric" in i for i in issues)


# ---- goal checks ----------------------------------------------------------

def test_target_duration_off(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 5}], project,
        goal={"target_duration_sec": 30})
    assert any("deviates from target" in i for i in issues)


def test_target_duration_within_tolerance(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 30}], project,
        goal={"target_duration_sec": 30})
    assert issues == []


def test_must_include_keyword_missing(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 5}], project,
        goal={"must_include_keywords": ["espresso"]})
    assert any("required keyword" in i and "espresso" in i for i in issues)


def test_must_include_keyword_present(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 25, "end": 40}], project,
        goal={"must_include_keywords": ["espresso"]})
    assert not any("required keyword" in i for i in issues)


def test_must_exclude_keyword_trapped(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 40, "end": 60}], project,
        goal={"must_exclude_keywords": ["weather"]})
    assert any("forbidden keyword" in i and "weather" in i for i in issues)


def test_min_clips(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 5}], project,
        goal={"min_clips": 3})
    assert any("at least 3" in i for i in issues)


def test_max_clips(project):
    issues = validators.validate_clips(
        [{"video_id": "v1", "start": 0, "end": 2},
         {"video_id": "v1", "start": 3, "end": 5},
         {"video_id": "v1", "start": 6, "end": 8}],
        project, goal={"max_clips": 2})
    assert any("at most 2" in i for i in issues)
