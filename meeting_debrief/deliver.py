"""Deliver extracted feature requests to GitHub issues and/or Mentat.

Two independent delivery legs over the feature-requests JSON that
extract.py writes:

  Leg A - GitHub: one ``gh issue create`` subprocess per feature request.
  Leg B - Mentat: one ``mentat.record_capture`` MCP tool call per feature
    request over streamable HTTP (official ``mcp`` SDK, client half),
    authenticated with a service-account bearer token.

Failure model: one leg failing never blocks the other, and one item
failing never blocks the next item. Every attempt lands in a per-item
results table; nothing raises mid-batch. Exit code 0 only when every
requested delivery succeeded.

Env (Leg B): MENTAT_MCP_URL, MENTAT_SERVICE_TOKEN,
MENTAT_EXPERT_TEAM_MEMBER_ID. Resolved through the package's .env
loading, same as transcription API keys.

Usage:
  python -m meeting_debrief.deliver feature_requests.json \
      [--github-repo owner/name] [--mentat] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys

# One `gh issue create` is a single HTTP round-trip; 120s is a network
# timeout for a hung gh process, not a tuning knob.
_GH_TIMEOUT_S = 120

# Mentat caps the capture topic column at 200 characters.
_TOPIC_MAX_CHARS = 200

_RECORD_CAPTURE_QUESTION = "What feature was requested in the meeting and why?"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_issue_body(fr: dict, captured_on: str | None = None) -> str:
    """Render one feature request as markdown.

    Used verbatim as both the GitHub issue body and the
    `mentat.record_capture` answer. Sections with no content are omitted
    entirely, never rendered as empty headings.
    """
    lines: list[str] = []

    def section(heading: str, content_lines: list[str]) -> None:
        if not content_lines:
            return
        if lines:
            lines.append("")
        lines.append(f"## {heading}")
        lines.append("")
        lines.extend(content_lines)

    problem = (fr.get("problem") or "").strip()
    if problem:
        section("Problem", [problem])

    detail = (fr.get("discussed_detail") or "").strip()
    if detail:
        section("Discussed detail", [detail])

    section("Decisions", [f"- {d}" for d in fr.get("decisions") or []])
    section("Open questions", [f"- {q}" for q in fr.get("open_questions") or []])
    section(
        "Acceptance criteria",
        [f"- {a}" for a in fr.get("acceptance_criteria") or []],
    )

    quote_lines: list[str] = []
    for q in fr.get("quotes") or []:
        if quote_lines:
            quote_lines.append("")  # blank line separates blockquotes
        quote_lines.append(f"> [{q['ts']}] {q['speaker']}: {q['text']}")
    section("Source quotes", quote_lines)

    date = captured_on or datetime.date.today().isoformat()
    participants = ", ".join(fr.get("participants") or []) or "-"
    footer = (
        f"Captured from meeting by meet-capture pipeline on {date}"
        f" · proposer: {fr.get('proposer') or '-'}"
        f" · participants: {participants}"
    )
    if lines:
        lines.extend(["", "---", ""])
    lines.append(footer)
    return "\n".join(lines) + "\n"


def _row(leg: str, title: str, ok: bool, detail: str) -> dict:
    return {"leg": leg, "title": title, "ok": ok, "detail": detail}


# ---------------------------------------------------------------------------
# Leg A - GitHub issues via gh
# ---------------------------------------------------------------------------

def deliver_github(
    feature_requests: list[dict], repo: str, dry_run: bool = False
) -> list[dict]:
    """Create one GitHub issue per feature request. Never raises."""
    results: list[dict] = []
    for fr in feature_requests:
        title = fr.get("title") or "(untitled)"
        body = render_issue_body(fr)

        if dry_run:
            print(f"\n--- [dry-run] gh issue create --repo {repo} ---")
            print(f"Title: {title}")
            print(body, end="")
            results.append(_row("github", title, True, "dry-run (not sent)"))
            continue

        try:
            proc = subprocess.run(
                ["gh", "issue", "create", "--repo", repo,
                 "--title", title, "--body", body],
                capture_output=True,
                text=True,
                timeout=_GH_TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001 - per-item isolation is the contract
            results.append(_row("github", title, False, f"{type(e).__name__}: {e}"))
            continue

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            results.append(
                _row("github", title, False, err or f"gh exited {proc.returncode}")
            )
            continue

        # gh prints the created issue URL on stdout (last non-empty line).
        stdout_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        url = next(
            (ln for ln in reversed(stdout_lines) if ln.startswith("http")),
            stdout_lines[-1] if stdout_lines else "created (no URL in gh output)",
        )
        results.append(_row("github", title, True, url))
    return results


# ---------------------------------------------------------------------------
# Leg B - mentat.record_capture over MCP (streamable HTTP)
# ---------------------------------------------------------------------------

async def _record_capture_async(url: str, token: str, arguments: dict):
    """One-shot MCP session: connect, initialize, call the tool, tear down.

    Import paths verified against mcp 1.27.1: the module is
    ``mcp.client.streamable_http`` and its context manager yields
    (read_stream, write_stream, get_session_id).
    """
    # Lazy import: Leg A and --dry-run must work without the mcp package.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(url, headers=headers) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool("mentat.record_capture", arguments)


def _record_capture(url: str, token: str, arguments: dict):
    """Sync wrapper: a fresh connection per item, so one failure cannot
    poison the next call's transport. Also the mocking seam for tests."""
    return asyncio.run(_record_capture_async(url, token, arguments))


def _result_error_text(result) -> str:
    """Extract the error text blocks from a CallToolResult with isError."""
    parts = [
        getattr(block, "text", "")
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "type", "") == "text"
    ]
    return "; ".join(p for p in parts if p) or "tool returned isError with no text"


def _parse_tool_result(result) -> dict:
    """Normalize a CallToolResult into the ResponseEnvelope dict.

    Mentat returns the envelope as structuredContent when available,
    otherwise as a JSON string inside the first text content block.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", "") == "text":
            return json.loads(block.text)
    raise ValueError("tool result had neither structuredContent nor a text block")


def _envelope_error(envelope: dict) -> str | None:
    """Return the envelope's error string, or None when it reports success."""
    err = envelope.get("error")
    if err:
        return err if isinstance(err, str) else json.dumps(err)
    if envelope.get("ok") is False or envelope.get("status") in ("error", "failed"):
        return json.dumps(envelope)
    return None


def build_capture_arguments(fr: dict, expert_team_member_id: str | None) -> dict:
    """Arguments for one mentat.record_capture call."""
    title = fr.get("title") or "(untitled)"
    return {
        "expert_team_member_id": expert_team_member_id,
        "topic": ("Feature request: " + title)[:_TOPIC_MAX_CHARS],
        "question": _RECORD_CAPTURE_QUESTION,
        "answer": render_issue_body(fr),
    }


def deliver_mentat(feature_requests: list[dict], dry_run: bool = False) -> list[dict]:
    """Record one capture per feature request in Mentat. Never raises."""
    results: list[dict] = []
    url = os.environ.get("MENTAT_MCP_URL")
    token = os.environ.get("MENTAT_SERVICE_TOKEN")
    member_id = os.environ.get("MENTAT_EXPERT_TEAM_MEMBER_ID")

    missing = [
        name
        for name, value in (
            ("MENTAT_MCP_URL", url),
            ("MENTAT_SERVICE_TOKEN", token),
            ("MENTAT_EXPERT_TEAM_MEMBER_ID", member_id),
        )
        if not value
    ]
    if missing and not dry_run:
        err = "missing env: " + ", ".join(missing)
        return [
            _row("mentat", fr.get("title") or "(untitled)", False, err)
            for fr in feature_requests
        ]

    for fr in feature_requests:
        title = fr.get("title") or "(untitled)"
        arguments = build_capture_arguments(fr, member_id)

        if dry_run:
            print(f"\n--- [dry-run] mentat.record_capture -> {url or '(MENTAT_MCP_URL unset)'} ---")
            print(f"expert_team_member_id: {member_id or '(unset)'}")
            print(f"topic: {arguments['topic']}")
            print(f"question: {arguments['question']}")
            print("answer:")
            print(arguments["answer"], end="")
            results.append(_row("mentat", title, True, "dry-run (not sent)"))
            continue

        try:
            result = _record_capture(url, token, arguments)
            if getattr(result, "isError", False):
                raise RuntimeError(_result_error_text(result))
            envelope = _parse_tool_result(result)
        except Exception as e:  # noqa: BLE001 - per-item isolation is the contract
            results.append(_row("mentat", title, False, f"{type(e).__name__}: {e}"))
            continue

        env_err = _envelope_error(envelope)
        if env_err:
            results.append(_row("mentat", title, False, env_err))
            continue

        ref = envelope.get("capture_id") or envelope.get("id")
        detail = f"recorded (id={ref})" if ref else "recorded"
        results.append(_row("mentat", title, True, detail))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_feature_requests(path: str) -> list[dict]:
    """Load the feature-requests JSON: a bare list (extract.py's output)
    or a {"feature_requests": [...]} wrapper."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("feature_requests"), list):
        data = data["feature_requests"]
    if not isinstance(data, list) or not all(isinstance(fr, dict) for fr in data):
        raise ValueError(
            f"{path}: expected a JSON list of feature-request objects "
            "(or a {'feature_requests': [...]} wrapper)"
        )
    return data


def print_results(results: list[dict]) -> None:
    print("\nDelivery results:")
    title_w = min(max((len(r["title"]) for r in results), default=5), 48)
    for r in results:
        status = "ok" if r["ok"] else "FAIL"
        title = r["title"]
        if len(title) > title_w:
            title = title[: title_w - 3] + "..."
        detail = " ".join(str(r["detail"]).split())  # keep the table one line per item
        print(f"  {r['leg']:<7} {status:<5} {title:<{title_w}}  {detail}")
    ok_count = sum(1 for r in results if r["ok"])
    print(f"\n{ok_count}/{len(results)} deliveries succeeded")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m meeting_debrief.deliver",
        description=(
            "Deliver extracted feature requests to GitHub issues (via gh) "
            "and/or Mentat (mentat.record_capture over MCP)."
        ),
    )
    parser.add_argument(
        "feature_requests_json",
        help="feature-requests JSON (list of objects, as written by extract.py)",
    )
    parser.add_argument(
        "--github-repo",
        default=None,
        metavar="OWNER/NAME",
        help="create one GitHub issue per feature request via the gh CLI",
    )
    parser.add_argument(
        "--mentat",
        action="store_true",
        help=(
            "record one mentat.record_capture per feature request "
            "(needs MENTAT_MCP_URL, MENTAT_SERVICE_TOKEN, "
            "MENTAT_EXPERT_TEAM_MEMBER_ID)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be sent; no subprocess or network calls",
    )
    args = parser.parse_args()

    if not args.github_repo and not args.mentat:
        parser.error("nothing to do: pass --github-repo and/or --mentat")

    # Same .env resolution as the transcription API keys.
    from meeting_debrief.cli import _load_dotenv
    _load_dotenv()

    try:
        feature_requests = load_feature_requests(args.feature_requests_json)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not feature_requests:
        print("No feature requests in input; nothing to deliver.")
        sys.exit(0)

    results: list[dict] = []
    if args.github_repo:
        results.extend(
            deliver_github(feature_requests, args.github_repo, dry_run=args.dry_run)
        )
    if args.mentat:
        results.extend(deliver_mentat(feature_requests, dry_run=args.dry_run))

    print_results(results)
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
