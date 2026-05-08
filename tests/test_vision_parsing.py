"""Vision LLMs return wonky output (markdown fences, prose prefixes,
flat-out refusals). Make sure our parser is robust."""

import pytest

import vision


@pytest.mark.parametrize("raw,expected_keys", [
    # plain JSON
    ('{"subject":"cat","scene":"indoor","interest":7}',
        {"subject", "scene", "interest"}),
    # markdown fence
    ('```json\n{"subject":"dog","interest":5}\n```',
        {"subject", "interest"}),
    # bare ``` fence (no `json` tag)
    ('```\n{"subject":"sky","interest":4}\n```',
        {"subject", "interest"}),
    # prose prefix
    ('Sure, here is your analysis: {"subject":"food","interest":8}',
        {"subject", "interest"}),
    # trailing commentary
    ('{"subject":"sky","interest":3} — let me know if you want more.',
        {"subject", "interest"}),
    # nested object
    ('{"subject":"x","tags":["a","b"],"meta":{"foo":1}}',
        {"subject", "tags", "meta"}),
])
def test_parses_valid_payloads(raw, expected_keys):
    parsed = vision._parse_vision_json(raw)
    assert "_parse_error" not in parsed
    for k in expected_keys:
        assert k in parsed


def test_empty_response_flagged():
    parsed = vision._parse_vision_json("")
    assert "_parse_error" in parsed


def test_no_json_in_reply_flagged():
    parsed = vision._parse_vision_json("I cannot tag this image safely.")
    assert "_parse_error" in parsed


def test_unbalanced_braces_flagged():
    parsed = vision._parse_vision_json('{"subject": "cat"')
    assert "_parse_error" in parsed


# ---- frame_search_text ----------------------------------------------------

def test_search_text_full_frame():
    frame = {"tags": {
        "subject": "cat close-up", "scene": "indoor",
        "action": "still", "interest": 7,
        "tags": ["cute", "pet"], "note": "orange tabby",
    }}
    text = vision.frame_search_text(frame)
    assert "cat close-up" in text
    assert "indoor" in text
    assert "cute" in text
    assert text == text.lower()


def test_search_text_handles_missing_fields():
    assert vision.frame_search_text({}) == ""
    assert vision.frame_search_text({"tags": {}}) == ""
    # api_error should yield empty (don't pollute search results)
    assert vision.frame_search_text({"tags": {"_api_error": "timeout"}}) == ""
