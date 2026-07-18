"""Tests for meeting_debrief.extract (no network: OpenAI is mocked)."""

import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meeting_debrief.extract import (  # noqa: E402
    apply_speaker_map,
    degradation_floor,
    extract_feature_requests,
    load_segments,
    parse_speaker_map_arg,
    parse_speaker_transcript,
    render_markdown,
    verify_quotes,
)

SAMPLE_TXT = """\
[00:32] Speaker A: Hello again.
[10:27] Speaker A: the user chats via Slack, they make feature requests.
[12:03] Speaker B: to me it feels almost like a setting, a default route.
"""

FULL_FR = {
    "title": "Default-route setting for feature requests",
    "kind": "feature",
    "problem": "Requests need routing to Drydock without manual triage.",
    "discussed_detail": "All new feature requests go routed to Drydock by default.",
    "proposer": "Speaker B",
    "participants": ["Speaker A", "Speaker B"],
    "decisions": ["Use a default-route setting for the pilot"],
    "open_questions": ["How does multi-destination routing work later?"],
    "acceptance_criteria": [],
    "quotes": [
        {"ts": "12:03", "speaker": "Speaker B", "text": "to me it feels almost like a setting"}
    ],
}


# --- parsing ---------------------------------------------------------------


def test_parse_speaker_transcript(tmp_path):
    p = tmp_path / "x_transcript_speakers.txt"
    p.write_text(SAMPLE_TXT)
    segments = parse_speaker_transcript(str(p))
    assert len(segments) == 3
    assert segments[0] == {
        "start": 32.0, "end": 32.0, "speaker": "Speaker A", "text": "Hello again."
    }
    assert segments[1]["start"] == 627.0  # 10:27


def test_parse_rejects_malformed_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("not a transcript line\n")
    with pytest.raises(ValueError, match="does not match"):
        parse_speaker_transcript(str(p))


def test_parse_rejects_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no transcript lines"):
        parse_speaker_transcript(str(p))


def test_load_segments_json_validates_keys(tmp_path):
    p = tmp_path / "segs.json"
    p.write_text(json.dumps([{"start": 1.0, "text": "hi"}]))  # missing speaker
    with pytest.raises(ValueError, match="missing keys"):
        load_segments(str(p))


def test_load_segments_json_roundtrip(tmp_path):
    segs = [{"start": 5.0, "end": 6.0, "speaker": "Karthik", "text": "hello"}]
    p = tmp_path / "segs.json"
    p.write_text(json.dumps(segs))
    assert load_segments(str(p)) == segs


# --- speaker mapping -------------------------------------------------------


def test_speaker_map_arg_and_apply():
    mapping = parse_speaker_map_arg("Speaker A=Karthik, Speaker B=Alex")
    assert mapping == {"Speaker A": "Karthik", "Speaker B": "Alex"}
    segs = [{"start": 0.0, "end": 0.0, "speaker": "Speaker A", "text": "hi"},
            {"start": 1.0, "end": 1.0, "speaker": "Speaker C", "text": "yo"}]
    out = apply_speaker_map(segs, mapping)
    assert out[0]["speaker"] == "Karthik"
    assert out[1]["speaker"] == "Speaker C"  # unmapped label passes through


def test_speaker_map_arg_rejects_garbage():
    with pytest.raises(ValueError, match="not 'Label=Name'"):
        parse_speaker_map_arg("SpeakerA-Karthik")


# --- extraction call (mocked) ---------------------------------------------


SWEEP_PAYLOAD = {
    "candidates": [
        {"title": "Default-route setting", "kind": "feature", "timestamps": ["12:03"]},
        {"title": "Slack error on onboarding", "kind": "bug", "timestamps": ["03:10"]},
    ]
}


def _mock_openai_returning(*payloads: dict):
    """Mock OpenAI module; chat.completions.create returns payloads in order.

    The last payload repeats if called more times (single-payload callers
    from the one-pass era keep working).
    """
    responses = [
        mock.Mock(choices=[mock.Mock(message=mock.Mock(content=json.dumps(p)))])
        for p in payloads
    ]
    client = mock.Mock()
    client.chat.completions.create.side_effect = responses[:-1] + [responses[-1]] * 10
    openai_mod = mock.Mock()
    openai_mod.OpenAI.return_value = client
    return openai_mod, client


def test_extract_happy_path_two_pass(monkeypatch):
    payload = {"feature_requests": [FULL_FR]}
    openai_mod, client = _mock_openai_returning(SWEEP_PAYLOAD, payload)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    segs = [{"start": 0.0, "end": 1.0, "speaker": "Speaker A", "text": "hi"}]
    result = extract_feature_requests(segs, model="test-model")
    assert result["feature_requests"] == payload["feature_requests"]
    # Sweep candidates surfaced for recall debugging
    assert result["candidates"] == SWEEP_PAYLOAD["candidates"]

    assert client.chat.completions.create.call_count == 2
    sweep_kwargs = client.chat.completions.create.call_args_list[0].kwargs
    detail_kwargs = client.chat.completions.create.call_args_list[1].kwargs

    # Pass 1: sweep uses the candidate schema, strict structured outputs
    assert sweep_kwargs["model"] == "test-model"
    assert sweep_kwargs["response_format"]["type"] == "json_schema"
    assert sweep_kwargs["response_format"]["json_schema"]["strict"] is True
    assert sweep_kwargs["response_format"]["json_schema"]["name"] == "candidates"
    assert "[00:00] Speaker A: hi" in sweep_kwargs["messages"][1]["content"]

    # Pass 2: detail pass gets the full schema AND the candidate list
    assert detail_kwargs["response_format"]["json_schema"]["name"] == "feature_requests"
    assert detail_kwargs["response_format"]["json_schema"]["strict"] is True
    detail_user = detail_kwargs["messages"][1]["content"]
    assert "[00:00] Speaker A: hi" in detail_user  # full transcript still present
    assert "Default-route setting" in detail_user
    assert "Slack error on onboarding" in detail_user


def test_extract_fails_on_empty_sweep(monkeypatch):
    openai_mod, client = _mock_openai_returning({"candidates": []})
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(RuntimeError, match="zero candidates"):
        extract_feature_requests(
            [{"start": 0.0, "end": 0.0, "speaker": "A", "text": "x"}]
        )
    assert client.chat.completions.create.call_count == 1  # never reached pass 2


def test_extract_requires_api_key(monkeypatch):
    openai_mod, _ = _mock_openai_returning({})
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        extract_feature_requests(
            [{"start": 0.0, "end": 0.0, "speaker": "A", "text": "x"}]
        )


def test_extract_rejects_oversized_transcript(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    big = [{"start": float(i), "end": float(i), "speaker": "A", "text": "x" * 1000}
           for i in range(250)]
    with pytest.raises(ValueError, match="cap"):
        extract_feature_requests(big)


def test_extract_rejects_unexpected_shape(monkeypatch):
    # Valid sweep, then a detail pass that returns the wrong shape.
    openai_mod, _ = _mock_openai_returning(SWEEP_PAYLOAD, {"wrong": []})
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(RuntimeError, match="unexpected shape"):
        extract_feature_requests(
            [{"start": 0.0, "end": 0.0, "speaker": "A", "text": "x"}]
        )


# --- quote verification (code gate, not prompt) ----------------------------


def test_verify_quotes_keeps_verbatim_drops_paraphrase():
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "A",
         "text": "the user chats via Slack, they make feature requests."},
    ]
    frs = [{
        **FULL_FR,
        "quotes": [
            # verbatim contiguous span (case/punctuation-insensitive)
            {"ts": "00:00", "speaker": "A", "text": "chats via Slack, they make"},
            # paraphrase - must be dropped
            {"ts": "00:00", "speaker": "A", "text": "users talk in Slack to request features"},
            # empty - must be dropped
            {"ts": "00:00", "speaker": "A", "text": ""},
        ],
    }]
    kept, dropped = verify_quotes(frs, segs)
    assert (kept, dropped) == (1, 2)
    assert [q["text"] for q in frs[0]["quotes"]] == ["chats via Slack, they make"]


def test_verify_quotes_spans_segment_boundary():
    # Normalized transcript is joined with spaces; a quote crossing two
    # consecutive segments of the same utterance still verifies.
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "we should be able"},
        {"start": 1.0, "end": 2.0, "speaker": "A", "text": "to have a meeting"},
    ]
    frs = [{**FULL_FR, "quotes": [
        {"ts": "00:00", "speaker": "A", "text": "should be able to have"},
    ]}]
    kept, dropped = verify_quotes(frs, segs)
    assert (kept, dropped) == (1, 0)


def test_unaccounted_candidates():
    from meeting_debrief.extract import unaccounted_candidates
    cands = [
        {"title": "kept", "kind": "feature", "timestamps": ["12:03", "15:00"]},
        {"title": "dropped", "kind": "bug", "timestamps": ["20:09"]},
    ]
    frs = [{**FULL_FR}]  # quotes at 12:03 only
    missing = unaccounted_candidates(cands, frs)
    assert [c["title"] for c in missing] == ["dropped"]


# --- rendering -------------------------------------------------------------


def test_render_markdown_full():
    md = render_markdown([FULL_FR], "call.json")
    assert "## 1. Default-route setting for feature requests" in md
    assert "### Decisions" in md
    assert "### Open questions" in md
    assert "### Acceptance criteria" not in md  # empty list -> section omitted
    assert "> [12:03] Speaker B: to me it feels almost like a setting" in md


def test_render_markdown_minimal_omits_empty_sections():
    minimal = {**FULL_FR, "decisions": [], "open_questions": [], "quotes": []}
    md = render_markdown([minimal], "x")
    assert "### Decisions" not in md
    assert "### Open questions" not in md
    assert "### Source quotes" not in md
    assert "### Problem" in md


# --- degradation floor -----------------------------------------------------


def test_degradation_floor_groups_blocks(tmp_path):
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "first block"},
        {"start": 10.0, "end": 11.0, "speaker": "B", "text": "same block"},
        {"start": 120.0, "end": 121.0, "speaker": "A", "text": "second block"},  # >45s gap
    ]
    out_base = str(tmp_path / "call")
    path = degradation_floor(segs, RuntimeError("api down"), out_base)
    assert path.endswith("_feature_requests_raw.md")
    content = open(path).read()
    assert "UNPROCESSED" in content
    assert "RuntimeError: api down" in content
    assert "## Discussion block 1" in content
    assert "## Discussion block 2" in content
    assert "[02:00] A: second block" in content


def test_degradation_floor_never_raises(tmp_path):
    # Even with a hostile error message, the floor writes and returns.
    path = degradation_floor(
        [{"start": 0.0, "end": 0.0, "speaker": "A", "text": "x"}],
        ValueError("weird \x00 error"),
        str(tmp_path / "y"),
    )
    assert os.path.exists(path)
