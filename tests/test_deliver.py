"""Tests for meeting_debrief.deliver — no network, no real gh calls."""

import json
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from meeting_debrief import deliver


FULL = {
    "title": "Add Confluence export",
    "problem": "Debrief reports are trapped in local markdown files.",
    "discussed_detail": "Karthik wants reports pushed to the team wiki.\n\nAgreed on a one-way sync.",
    "proposer": "Karthik",
    "participants": ["Karthik", "Alex"],
    "decisions": ["One-way sync only", "Markdown stays the source of truth"],
    "open_questions": ["Which Confluence space?"],
    "acceptance_criteria": ["Report appears in Confluence within 5 minutes"],
    "quotes": [
        {"ts": "12:34", "speaker": "Karthik", "text": "Can this land in our wiki?"},
        {"ts": "13:01", "speaker": "Alex", "text": "One-way sync is a day of work."},
    ],
}

MINIMAL = {
    "title": "Tiny ask",
    "problem": "Something is missing.",
    "discussed_detail": "",
    "proposer": "Alex",
    "participants": [],
    "decisions": [],
    "open_questions": [],
    "acceptance_criteria": [],
    "quotes": [],
}


def _fake_tool_result(envelope=None, is_error=False, content=None):
    return SimpleNamespace(
        isError=is_error,
        structuredContent=envelope,
        content=content or [],
    )


def _run_main(argv):
    with mock.patch.object(sys, "argv", ["deliver"] + argv):
        with pytest.raises(SystemExit) as exc_info:
            deliver.main()
    return exc_info.value.code


# ---------------------------------------------------------------------------
# (a) markdown body rendering
# ---------------------------------------------------------------------------

def test_render_issue_body_full_schema():
    body = deliver.render_issue_body(FULL, captured_on="2026-07-17")

    # All sections present, in order.
    headings = [
        "## Problem",
        "## Discussed detail",
        "## Decisions",
        "## Open questions",
        "## Acceptance criteria",
        "## Source quotes",
    ]
    positions = [body.index(h) for h in headings]
    assert positions == sorted(positions)

    assert "Debrief reports are trapped in local markdown files." in body
    assert "Agreed on a one-way sync." in body
    assert "- One-way sync only" in body
    assert "- Markdown stays the source of truth" in body
    assert "- Which Confluence space?" in body
    assert "- Report appears in Confluence within 5 minutes" in body
    assert "> [12:34] Karthik: Can this land in our wiki?" in body
    assert "> [13:01] Alex: One-way sync is a day of work." in body

    assert (
        "Captured from meeting by meet-capture pipeline on 2026-07-17"
        " · proposer: Karthik · participants: Karthik, Alex"
    ) in body


def test_render_issue_body_minimal_omits_empty_sections():
    body = deliver.render_issue_body(MINIMAL, captured_on="2026-07-17")

    assert "## Problem" in body
    assert "Something is missing." in body
    # Empty string and empty lists: sections omitted, not rendered empty.
    assert "## Discussed detail" not in body
    assert "## Decisions" not in body
    assert "## Open questions" not in body
    assert "## Acceptance criteria" not in body
    assert "## Source quotes" not in body

    assert (
        "Captured from meeting by meet-capture pipeline on 2026-07-17"
        " · proposer: Alex · participants: -"
    ) in body


# ---------------------------------------------------------------------------
# (b) per-item isolation
# ---------------------------------------------------------------------------

def test_github_first_item_fails_second_still_delivered(tmp_path, capsys):
    path = tmp_path / "frs.json"
    path.write_text(json.dumps([FULL, MINIMAL]))

    fail = SimpleNamespace(returncode=1, stdout="", stderr="gh: boom")
    ok = SimpleNamespace(
        returncode=0,
        stdout="https://github.com/owner/name/issues/7\n",
        stderr="",
    )
    with mock.patch.object(
        deliver.subprocess, "run", side_effect=[fail, ok]
    ) as run_mock:
        code = _run_main([str(path), "--github-repo", "owner/name"])

    assert code != 0
    assert run_mock.call_count == 2  # second item attempted despite first failing
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "gh: boom" in out
    assert "https://github.com/owner/name/issues/7" in out


def test_mentat_first_item_fails_second_still_delivered(monkeypatch):
    monkeypatch.setenv("MENTAT_MCP_URL", "https://mcp-dev.example.test")
    monkeypatch.setenv("MENTAT_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("MENTAT_EXPERT_TEAM_MEMBER_ID", "42")

    with mock.patch.object(
        deliver,
        "_record_capture",
        side_effect=[
            ConnectionError("stream reset"),
            _fake_tool_result(envelope={"capture_id": "cap_9"}),
        ],
    ) as rc_mock:
        results = deliver.deliver_mentat([FULL, MINIMAL])

    assert rc_mock.call_count == 2
    assert [r["ok"] for r in results] == [False, True]
    assert "stream reset" in results[0]["detail"]
    assert results[1]["detail"] == "recorded (id=cap_9)"


def test_one_leg_failing_does_not_block_the_other(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MENTAT_MCP_URL", "https://mcp-dev.example.test")
    monkeypatch.setenv("MENTAT_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("MENTAT_EXPERT_TEAM_MEMBER_ID", "42")
    path = tmp_path / "frs.json"
    path.write_text(json.dumps([MINIMAL]))

    gh_fail = SimpleNamespace(returncode=1, stdout="", stderr="gh: no auth")
    with mock.patch.object(deliver.subprocess, "run", return_value=gh_fail), \
         mock.patch.object(
             deliver,
             "_record_capture",
             return_value=_fake_tool_result(envelope={"capture_id": "cap_1"}),
         ) as rc_mock:
        code = _run_main([str(path), "--github-repo", "owner/name", "--mentat"])

    assert code != 0  # github leg failed
    assert rc_mock.call_count == 1  # mentat leg still ran
    out = capsys.readouterr().out
    assert "gh: no auth" in out
    assert "recorded (id=cap_1)" in out
    assert "1/2 deliveries succeeded" in out


# ---------------------------------------------------------------------------
# (c) dry-run: zero subprocess / network calls
# ---------------------------------------------------------------------------

def test_dry_run_makes_no_subprocess_or_network_calls(tmp_path, capsys):
    path = tmp_path / "frs.json"
    path.write_text(json.dumps([FULL]))

    with mock.patch.object(deliver.subprocess, "run") as run_mock, \
         mock.patch.object(deliver, "_record_capture") as rc_mock:
        code = _run_main(
            [str(path), "--github-repo", "owner/name", "--mentat", "--dry-run"]
        )

    assert code == 0
    run_mock.assert_not_called()
    rc_mock.assert_not_called()

    out = capsys.readouterr().out
    # Full bodies are printed for both legs.
    assert out.count("## Problem") == 2
    assert "Title: Add Confluence export" in out
    assert "topic: Feature request: Add Confluence export" in out
    assert "question: What feature was requested in the meeting and why?" in out
    assert "2/2 deliveries succeeded" in out


# ---------------------------------------------------------------------------
# (d) topic 200-char cap
# ---------------------------------------------------------------------------

def test_topic_capped_at_200_chars(monkeypatch):
    monkeypatch.setenv("MENTAT_MCP_URL", "https://mcp-dev.example.test")
    monkeypatch.setenv("MENTAT_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("MENTAT_EXPERT_TEAM_MEMBER_ID", "42")
    long_title = "x" * 300
    fr = dict(MINIMAL, title=long_title)

    with mock.patch.object(
        deliver,
        "_record_capture",
        return_value=_fake_tool_result(envelope={"capture_id": "cap_2"}),
    ) as rc_mock:
        results = deliver.deliver_mentat([fr])

    assert results[0]["ok"]
    arguments = rc_mock.call_args.args[2]
    assert arguments["topic"] == ("Feature request: " + long_title)[:200]
    assert len(arguments["topic"]) == 200
    assert arguments["question"] == "What feature was requested in the meeting and why?"
    assert arguments["expert_team_member_id"] == "42"


def test_short_topic_not_truncated():
    args = deliver.build_capture_arguments(MINIMAL, "42")
    assert args["topic"] == "Feature request: Tiny ask"


# ---------------------------------------------------------------------------
# Supporting behavior
# ---------------------------------------------------------------------------

def test_parse_tool_result_text_block_fallback():
    envelope = {"capture_id": "cap_3", "ok": True}
    result = _fake_tool_result(
        envelope=None,
        content=[SimpleNamespace(type="text", text=json.dumps(envelope))],
    )
    assert deliver._parse_tool_result(result) == envelope


def test_mentat_envelope_error_marks_item_failed(monkeypatch):
    monkeypatch.setenv("MENTAT_MCP_URL", "https://mcp-dev.example.test")
    monkeypatch.setenv("MENTAT_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("MENTAT_EXPERT_TEAM_MEMBER_ID", "42")

    with mock.patch.object(
        deliver,
        "_record_capture",
        return_value=_fake_tool_result(envelope={"error": "token expired"}),
    ):
        results = deliver.deliver_mentat([MINIMAL])

    assert results == [
        {"leg": "mentat", "title": "Tiny ask", "ok": False, "detail": "token expired"}
    ]


def test_mentat_missing_env_fails_every_item_without_raising(monkeypatch):
    for var in (
        "MENTAT_MCP_URL",
        "MENTAT_SERVICE_TOKEN",
        "MENTAT_EXPERT_TEAM_MEMBER_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    with mock.patch.object(deliver, "_record_capture") as rc_mock:
        results = deliver.deliver_mentat([FULL, MINIMAL])

    rc_mock.assert_not_called()
    assert [r["ok"] for r in results] == [False, False]
    assert all("missing env" in r["detail"] for r in results)


def test_requires_at_least_one_leg(tmp_path, capsys):
    path = tmp_path / "frs.json"
    path.write_text(json.dumps([MINIMAL]))
    code = _run_main([str(path)])
    assert code == 2  # argparse error
    assert "--github-repo and/or --mentat" in capsys.readouterr().err


def test_load_feature_requests_accepts_wrapper(tmp_path):
    path = tmp_path / "frs.json"
    path.write_text(json.dumps({"feature_requests": [MINIMAL]}))
    assert deliver.load_feature_requests(str(path)) == [MINIMAL]
