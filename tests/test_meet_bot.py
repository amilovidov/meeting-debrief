"""Tests for meeting_debrief.meet_bot — no network, httpx fully mocked."""

import json
from unittest import mock

import pytest

from meeting_debrief import meet_bot


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else (
            json.dumps(payload) if payload is not None else ""
        )

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


# Documented download_url shape: participant object + words with
# {relative, absolute} timestamps (absolute is null for async transcription).
NAMED_TRANSCRIPT = [
    {
        "participant": {
            "id": 100,
            "name": "Alexander Milovidov",
            "is_host": True,
            "platform": "google_meet",
            "extra_data": None,
            "email": None,
        },
        "language_code": "en",
        "words": [
            {
                "text": "Let's",
                "start_timestamp": {"relative": 5.0, "absolute": None},
                "end_timestamp": {"relative": 5.4, "absolute": None},
            },
            {
                "text": "start.",
                "start_timestamp": {"relative": 5.4, "absolute": None},
                "end_timestamp": {"relative": 6.1, "absolute": None},
            },
        ],
    },
    {
        "participant": {
            "id": 200,
            "name": "Karthik",
            "is_host": False,
            "platform": "google_meet",
            "extra_data": None,
            "email": None,
        },
        "language_code": "en",
        "words": [
            {
                "text": "Sounds",
                "start_timestamp": {"relative": 65.2, "absolute": None},
                "end_timestamp": {"relative": 65.6, "absolute": None},
            },
            {
                "text": "good.",
                "start_timestamp": {"relative": 65.6, "absolute": None},
                "end_timestamp": {"relative": 66.0, "absolute": None},
            },
        ],
    },
]

# Anonymous speakers, deliberately out of order, mixing the documented shape
# (participant with null name) and the legacy shape (speaker string + plain
# float start_time/end_time).
ANON_TRANSCRIPT = [
    {
        "speaker": "Speaker 1",
        "words": [{"text": "Hi", "start_time": 2.0, "end_time": 2.3}],
    },
    {
        "participant": {"id": 300, "name": None},
        "words": [
            {
                "text": "Hello",
                "start_timestamp": {"relative": 1.0, "absolute": None},
                "end_timestamp": {"relative": 1.5, "absolute": None},
            }
        ],
    },
]


@pytest.fixture
def recall_env(monkeypatch):
    monkeypatch.setenv("RECALL_API_KEY", "sekret")
    monkeypatch.delenv("RECALL_BASE", raising=False)


# ---------------------------------------------------------------------------
# (a) segments conversion
# ---------------------------------------------------------------------------

def test_segments_from_named_participants():
    segments = meet_bot.transcript_to_segments(NAMED_TRANSCRIPT)
    assert segments == [
        {
            "start": 5.0,
            "end": 6.1,
            "speaker": "Alexander Milovidov",
            "text": "Let's start.",
        },
        {"start": 65.2, "end": 66.0, "speaker": "Karthik", "text": "Sounds good."},
    ]


def test_segments_from_anonymous_speakers_sorted_and_labeled():
    segments = meet_bot.transcript_to_segments(ANON_TRANSCRIPT)
    # Sorted by start time despite reversed input order.
    assert segments == [
        {"start": 1.0, "end": 1.5, "speaker": "Speaker 300", "text": "Hello"},
        {"start": 2.0, "end": 2.3, "speaker": "Speaker 1", "text": "Hi"},
    ]


def test_segments_skip_empty_entries_and_unwrap_results_dict():
    wrapped = {"results": NAMED_TRANSCRIPT + [{"participant": {"id": 9}, "words": []}]}
    segments = meet_bot.transcript_to_segments(wrapped)
    assert len(segments) == 2


def test_segments_unrecognized_shape_raises():
    with pytest.raises(RuntimeError, match="Unrecognized transcript shape"):
        meet_bot.transcript_to_segments({"weird": True})


# ---------------------------------------------------------------------------
# (b) transcript_speakers.txt line format
# ---------------------------------------------------------------------------

def test_transcript_speakers_line_format_exact(tmp_path):
    segments = meet_bot.transcript_to_segments(NAMED_TRANSCRIPT)
    _, speakers_path = meet_bot.write_outputs(segments, str(tmp_path), "call1")

    content = open(speakers_path).read()
    assert content == (
        "[00:05] Alexander Milovidov: Let's start.\n"
        "[01:05] Karthik: Sounds good.\n"
    )
    assert speakers_path.endswith("call1_transcript_speakers.txt")


def test_segments_json_written(tmp_path):
    segments = meet_bot.transcript_to_segments(NAMED_TRANSCRIPT)
    segments_path, _ = meet_bot.write_outputs(segments, str(tmp_path), "call1")

    assert segments_path.endswith("call1_segments.json")
    with open(segments_path) as f:
        assert json.load(f) == segments


# ---------------------------------------------------------------------------
# (c) dispatch payload shape, both providers
# ---------------------------------------------------------------------------

def test_dispatch_payload_recallai_async(recall_env):
    with mock.patch.object(
        meet_bot.httpx, "post", return_value=FakeResponse({"id": "bot-1"})
    ) as post_mock:
        bot_id = meet_bot.dispatch("https://meet.google.com/abc-defg-hij")

    assert bot_id == "bot-1"
    assert post_mock.call_args.args[0] == "https://us-west-2.recall.ai/api/v1/bot/"
    body = post_mock.call_args.kwargs["json"]
    assert body["meeting_url"] == "https://meet.google.com/abc-defg-hij"
    # Per docs, recallai_async is NOT a Create Bot provider — the bot records
    # with the default config and transcription is requested post-call.
    assert "recording_config" not in body


def test_dispatch_payload_meeting_captions(recall_env):
    with mock.patch.object(
        meet_bot.httpx, "post", return_value=FakeResponse({"id": "bot-2"})
    ) as post_mock:
        bot_id = meet_bot.dispatch(
            "https://meet.google.com/abc-defg-hij", provider="meeting_captions"
        )

    assert bot_id == "bot-2"
    body = post_mock.call_args.kwargs["json"]
    assert body["recording_config"] == {
        "transcript": {"provider": {"meeting_captions": {}}}
    }


def test_dispatch_unknown_provider_raises(recall_env):
    with pytest.raises(ValueError, match="Unknown provider"):
        meet_bot.dispatch("https://meet.google.com/x", provider="deepgram")


def test_create_async_transcript_payload(recall_env):
    with mock.patch.object(
        meet_bot.httpx,
        "post",
        return_value=FakeResponse(
            {"id": "tr-1", "data": {"download_url": "https://bucket/t.json"}}
        ),
    ) as post_mock:
        transcript = meet_bot.request_async_transcript("rec-1")

    assert transcript["id"] == "tr-1"
    assert post_mock.call_args.args[0] == (
        "https://us-west-2.recall.ai/api/v1/recording/rec-1/create_transcript/"
    )
    assert post_mock.call_args.kwargs["json"] == {
        "provider": {"recallai_async": {"language_code": "auto"}},
        "diarization": {"use_separate_streams_when_available": True},
    }


# ---------------------------------------------------------------------------
# (d) auth header is Token-style, not Bearer
# ---------------------------------------------------------------------------

def test_auth_header_is_token_not_bearer(recall_env):
    headers = meet_bot.auth_headers()
    assert headers["Authorization"] == "Token sekret"
    assert not headers["Authorization"].startswith("Bearer")


def test_auth_headers_used_on_dispatch(recall_env):
    with mock.patch.object(
        meet_bot.httpx, "post", return_value=FakeResponse({"id": "bot-3"})
    ) as post_mock:
        meet_bot.dispatch("https://meet.google.com/x")
    assert post_mock.call_args.kwargs["headers"]["Authorization"] == "Token sekret"


def test_auth_headers_missing_key_raises(monkeypatch):
    monkeypatch.delenv("RECALL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RECALL_API_KEY not set"):
        meet_bot.auth_headers()


# ---------------------------------------------------------------------------
# Supporting behavior: polling, transcript retrieval paths, HTTP errors
# ---------------------------------------------------------------------------

def test_poll_prints_transitions_and_stops_on_done(recall_env, capsys):
    payloads = [
        {"status_changes": [{"code": "joining_call"}]},
        {
            "status_changes": [
                {"code": "joining_call"},
                {"code": "in_waiting_room"},
                {"code": "in_call_recording"},
            ]
        },
        {
            "status_changes": [
                {"code": "joining_call"},
                {"code": "in_waiting_room"},
                {"code": "in_call_recording"},
                {"code": "call_ended"},
                {"code": "done"},
            ]
        },
    ]
    with mock.patch.object(
        meet_bot.httpx, "get", side_effect=[FakeResponse(p) for p in payloads]
    ) as get_mock:
        bot = meet_bot.poll("bot-1", timeout_min=1, interval_s=0)

    assert get_mock.call_count == 3
    assert bot["status_changes"][-1]["code"] == "done"
    out = capsys.readouterr().out
    for code in ("joining_call", "in_waiting_room", "in_call_recording", "done"):
        assert out.count(code) == 1  # each transition printed exactly once


def test_poll_stops_on_fatal_with_sub_code(recall_env, capsys):
    payload = {
        "status_changes": [
            {"code": "joining_call"},
            {"code": "fatal", "sub_code": "meeting_not_found"},
        ]
    }
    with mock.patch.object(
        meet_bot.httpx, "get", return_value=FakeResponse(payload)
    ):
        bot = meet_bot.poll("bot-1", timeout_min=1, interval_s=0)

    assert bot["status_changes"][-1]["code"] == "fatal"
    assert "meeting_not_found" in capsys.readouterr().out


def test_fetch_transcript_via_media_shortcuts_without_auth_header(recall_env):
    bot = {
        "recordings": [
            {
                "id": "rec-1",
                "media_shortcuts": {
                    "transcript": {
                        "data": {"download_url": "https://bucket/presigned.json"}
                    }
                },
            }
        ]
    }
    with mock.patch.object(
        meet_bot.httpx, "get", return_value=FakeResponse(NAMED_TRANSCRIPT)
    ) as get_mock:
        raw = meet_bot.fetch_transcript(bot, provider="meeting_captions")

    assert raw == NAMED_TRANSCRIPT
    assert get_mock.call_args.args[0] == "https://bucket/presigned.json"
    # Presigned URL: no Authorization header may be sent.
    assert "headers" not in get_mock.call_args.kwargs


def test_fetch_transcript_async_path_creates_then_polls_then_downloads(recall_env):
    bot = {"recordings": [{"id": "rec-1", "media_shortcuts": {}}]}
    job = FakeResponse({"id": "tr-1", "status": {"code": "processing"}, "data": None})
    job_done = FakeResponse(
        {
            "id": "tr-1",
            "status": {"code": "done"},
            "data": {"download_url": "https://bucket/t.json"},
        }
    )
    download = FakeResponse(ANON_TRANSCRIPT)

    with mock.patch.object(meet_bot.httpx, "post", return_value=job) as post_mock, \
         mock.patch.object(
             meet_bot.httpx, "get", side_effect=[job_done, download]
         ) as get_mock:
        raw = meet_bot.fetch_transcript(bot, provider="recallai_async", interval_s=0)

    assert raw == ANON_TRANSCRIPT
    assert post_mock.call_args.args[0].endswith(
        "/api/v1/recording/rec-1/create_transcript/"
    )
    # Last GET is the presigned download, no auth header.
    assert get_mock.call_args.args[0] == "https://bucket/t.json"
    assert "headers" not in get_mock.call_args.kwargs


def test_fetch_transcript_meeting_captions_missing_artifact_raises(recall_env):
    bot = {"recordings": [{"id": "rec-1", "media_shortcuts": {}}]}
    with pytest.raises(RuntimeError, match="captions"):
        meet_bot.fetch_transcript(bot, provider="meeting_captions")


def test_fetch_transcript_no_recordings_raises(recall_env):
    with pytest.raises(RuntimeError, match="no recordings"):
        meet_bot.fetch_transcript({"recordings": []}, provider="recallai_async")


def test_http_error_raises_with_body(recall_env):
    with mock.patch.object(
        meet_bot.httpx,
        "post",
        return_value=FakeResponse(status_code=401, text='{"detail":"Invalid token."}'),
    ):
        with pytest.raises(RuntimeError, match="HTTP 401.*Invalid token"):
            meet_bot.dispatch("https://meet.google.com/x")


def test_basename_from_url():
    assert (
        meet_bot._basename_from_url("https://meet.google.com/abc-defg-hij")
        == "abc-defg-hij"
    )
    assert (
        meet_bot._basename_from_url("https://meet.google.com/abc-defg-hij?authuser=0")
        == "abc-defg-hij"
    )
    assert meet_bot._basename_from_url("https://meet.google.com/") == "meeting"
