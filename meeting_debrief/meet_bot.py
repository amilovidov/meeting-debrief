"""Google Meet capture via a Recall.ai bot — post-call transcript, no websockets.

Flow: dispatch a bot into the meeting, poll its status every 15s until it is
done (or fatal), then pull the transcript and convert it to the
meeting-debrief segments format ({start, end, speaker, text}).

Two transcription providers:

- ``recallai_async`` (default): the bot only records. After the call, we POST
  Create Async Transcript on the recording ($0.15/hr) and poll the transcript
  job until its download_url appears. Works regardless of whether the host
  enabled captions. Per docs.recall.ai (read 2026-07-17), this provider is NOT
  valid inside recording_config at Create Bot time — async transcription is a
  separate post-call request on the recording
  (POST /api/v1/recording/{id}/create_transcript/).
- ``meeting_captions``: configured at Create Bot time
  (recording_config.transcript.provider.meeting_captions). Uses Meet's own
  captions; free, but produces nothing if captions were never turned on.

Either way the transcript lands at
recordings[].media_shortcuts.transcript.data.download_url (or the async
transcript object's data.download_url) — a presigned URL fetched WITHOUT the
auth header.

Auth is ``Authorization: Token <RECALL_API_KEY>`` — literal "Token", not
Bearer. Regions have independent credentials; base URL comes from RECALL_BASE
(default https://us-west-2.recall.ai). Rate limits (Create Bot 120/min,
Retrieve Bot 300/min) make 15s polling comfortable.

CLI:
    python -m meeting_debrief.meet_bot <meeting-url> [-o outdir]
        [--provider meeting_captions|recallai_async] [--probe]
        [--timeout-min N] [--basename NAME]

--probe dispatches, polls, dumps raw_bot.json / raw_transcript.json, prints a
summary of what came back (speakers, event counts, per-speaker word counts)
and exits — the go/no-go gate for Meet before trusting the pipeline.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse

import httpx

from meeting_debrief.cli import _load_dotenv

DEFAULT_BASE = "https://us-west-2.recall.ai"
DEFAULT_PROVIDER = "recallai_async"
PROVIDERS = ("meeting_captions", "recallai_async")
POLL_INTERVAL_S = 15  # Retrieve Bot limit is 300/min; 15s is 4/min.
# Async transcription runs several times faster than realtime, so even the
# longest capture the default --timeout-min allows (90 min) finishes well
# inside this. Raise it if a real job ever hits the ceiling.
ASYNC_TRANSCRIPT_TIMEOUT_MIN = 30
TERMINAL_CODES = {"done", "fatal"}
# Fallback source for RECALL_API_KEY only — the Zoom integration keeps its
# key in this repo's .env (see docs/research/RECALL_INTEGRATION.md there).
AIFUND_ENV_PATH = "/Users/amilovidov/src/AIFund-Challenge/.env"


def _load_recall_key_fallback(path: str = AIFUND_ENV_PATH) -> None:
    """If RECALL_API_KEY is still unset, read exactly that one key from the
    AIFund-Challenge .env. Nothing else is read from that file."""
    if os.environ.get("RECALL_API_KEY"):
        return
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != "RECALL_API_KEY":
                continue
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["RECALL_API_KEY"] = value
            return


def recall_base() -> str:
    return os.environ.get("RECALL_BASE", DEFAULT_BASE).rstrip("/")


def auth_headers() -> dict:
    key = os.environ.get("RECALL_API_KEY")
    if not key:
        raise RuntimeError(
            "RECALL_API_KEY not set. Put it in .env (cwd or repo root) or "
            f"in {AIFUND_ENV_PATH}."
        )
    # Literal "Token", not "Bearer" — Recall rejects Bearer.
    return {"Authorization": f"Token {key}"}


def _check(resp, what: str) -> dict:
    """Raise loudly (with the response body) on any HTTP error."""
    if resp.status_code >= 400:
        raise RuntimeError(f"{what} failed: HTTP {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------

def dispatch(meeting_url: str, provider: str = DEFAULT_PROVIDER) -> str:
    """Create a bot for the meeting. Returns the bot id.

    meeting_captions is configured at create time; recallai_async is not a
    valid Create Bot provider (per docs) — the bot just records with the
    default config and transcription is requested post-call on the recording.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")
    body: dict = {"meeting_url": meeting_url}
    if provider == "meeting_captions":
        body["recording_config"] = {
            "transcript": {"provider": {"meeting_captions": {}}}
        }
    resp = httpx.post(
        f"{recall_base()}/api/v1/bot/", json=body, headers=auth_headers(), timeout=30
    )
    bot = _check(resp, "Create Bot")
    return bot["id"]


def poll(bot_id: str, timeout_min: int = 90, interval_s: float = POLL_INTERVAL_S) -> dict:
    """Poll Retrieve Bot until a terminal status (done/fatal) or timeout.

    Prints each status transition (joining_call, in_waiting_room,
    in_call_recording, call_ended, done, fatal, ...) as it appears.
    Returns the final bot payload.
    """
    deadline = time.time() + timeout_min * 60
    seen = 0
    while True:
        resp = httpx.get(
            f"{recall_base()}/api/v1/bot/{bot_id}/", headers=auth_headers(), timeout=30
        )
        bot = _check(resp, "Retrieve Bot")
        changes = bot.get("status_changes") or []
        for change in changes[seen:]:
            code = change.get("code", "?")
            line = f"  [{time.strftime('%H:%M:%S')}] {code}"
            if change.get("sub_code"):
                line += f" ({change['sub_code']})"
            if change.get("message"):
                line += f": {change['message']}"
            print(line, flush=True)
        seen = len(changes)
        last = changes[-1].get("code") if changes else None
        if last in TERMINAL_CODES:
            return bot
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out after {timeout_min} min waiting for bot {bot_id} "
                f"(last status: {last})."
            )
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# Transcript retrieval
# ---------------------------------------------------------------------------

def _download_url_from_bot(bot: dict) -> str | None:
    """recordings[].media_shortcuts.transcript.data.download_url, defensively."""
    for rec in bot.get("recordings") or []:
        shortcut = (rec.get("media_shortcuts") or {}).get("transcript") or {}
        url = (shortcut.get("data") or {}).get("download_url")
        if url:
            return url
    return None


def request_async_transcript(recording_id: str) -> dict:
    """POST Create Async Transcript on the recording. Returns the transcript
    object. On 409 (a transcript already exists) falls back to listing."""
    body = {
        "provider": {"recallai_async": {"language_code": "auto"}},
        "diarization": {"use_separate_streams_when_available": True},
    }
    resp = httpx.post(
        f"{recall_base()}/api/v1/recording/{recording_id}/create_transcript/",
        json=body,
        headers=auth_headers(),
        timeout=30,
    )
    if resp.status_code == 409:
        print(
            "  transcript already exists for this recording; reusing it",
            flush=True,
        )
        return _latest_transcript_for_recording(recording_id)
    return _check(resp, "Create Async Transcript")


def _latest_transcript_for_recording(recording_id: str) -> dict:
    resp = httpx.get(
        f"{recall_base()}/api/v1/transcript/",
        params={"recording_id": recording_id},
        headers=auth_headers(),
        timeout=30,
    )
    payload = _check(resp, "List Transcripts")
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not results:
        raise RuntimeError(
            f"No transcripts found for recording {recording_id} despite a 409 "
            "on create. See raw_bot.json."
        )
    return results[0]


def wait_for_async_transcript(
    transcript_id: str,
    timeout_min: int = ASYNC_TRANSCRIPT_TIMEOUT_MIN,
    interval_s: float = POLL_INTERVAL_S,
) -> str:
    """Poll Retrieve Transcript until data.download_url appears. Returns the URL."""
    deadline = time.time() + timeout_min * 60
    last_code = None
    while True:
        resp = httpx.get(
            f"{recall_base()}/api/v1/transcript/{transcript_id}/",
            headers=auth_headers(),
            timeout=30,
        )
        transcript = _check(resp, "Retrieve Transcript")
        url = (transcript.get("data") or {}).get("download_url")
        if url:
            return url
        code = str((transcript.get("status") or {}).get("code") or "?")
        if code != last_code:
            print(f"  transcript status: {code}", flush=True)
            last_code = code
        if "error" in code or "fail" in code:
            raise RuntimeError(
                f"Async transcription failed: {json.dumps(transcript)[:2000]}"
            )
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out after {timeout_min} min waiting for transcript "
                f"{transcript_id} (last status: {code})."
            )
        time.sleep(interval_s)


def fetch_transcript(
    bot: dict,
    provider: str = DEFAULT_PROVIDER,
    interval_s: float = POLL_INTERVAL_S,
):
    """Get the raw transcript JSON for a finished bot.

    meeting_captions: the transcript artifact is already on the bot payload.
    recallai_async: request transcription on the recording, wait for the job,
    then download. The download_url is presigned — fetched WITHOUT the auth
    header (S3 rejects requests carrying both auth mechanisms).
    """
    url = _download_url_from_bot(bot)
    if url is None:
        if provider == "meeting_captions":
            raise RuntimeError(
                "Bot finished but no transcript artifact on "
                "recordings[].media_shortcuts.transcript — with the "
                "meeting_captions provider this usually means captions were "
                "never enabled in the meeting. Full payload: raw_bot.json. "
                "Re-run with --provider recallai_async to transcribe the "
                "recording instead."
            )
        recordings = bot.get("recordings") or []
        if not recordings:
            raise RuntimeError(
                "Bot finished with no recordings — nothing to transcribe. "
                "Full payload: raw_bot.json."
            )
        transcript = request_async_transcript(recordings[0]["id"])
        url = (transcript.get("data") or {}).get("download_url")
        if not url:
            url = wait_for_async_transcript(transcript["id"], interval_s=interval_s)
    resp = httpx.get(url, timeout=120, follow_redirects=True)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Transcript download failed: HTTP {resp.status_code}: {resp.text[:2000]}"
        )
    return resp.json()


# ---------------------------------------------------------------------------
# Conversion to meeting-debrief segments
# ---------------------------------------------------------------------------

def _ts(value) -> float | None:
    """Timestamp from either shape: {'relative': 12.3, 'absolute': ...} or a
    plain number (legacy start_time/end_time)."""
    if isinstance(value, dict):
        rel = value.get("relative")
        return float(rel) if rel is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _speaker_label(entry: dict) -> str:
    participant = entry.get("participant") or {}
    name = participant.get("name")
    if name:
        return str(name)
    pid = participant.get("id")
    if pid is not None:
        return f"Speaker {pid}"
    if entry.get("speaker"):
        return str(entry["speaker"])
    if entry.get("speaker_id") is not None:
        return f"Speaker {entry['speaker_id']}"
    return "Unknown"


def transcript_to_segments(raw) -> list[dict]:
    """Convert a Recall transcript payload to [{start, end, speaker, text}].

    Handles the documented download_url shape (participant + words with
    {relative, absolute} timestamps) and the legacy shape (speaker string +
    words with plain float start_time/end_time). Times are relative seconds.
    """
    if isinstance(raw, dict):
        for key in ("results", "data", "transcript"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise RuntimeError(
            f"Unrecognized transcript shape ({type(raw).__name__}) — "
            "inspect raw_transcript.json."
        )

    segments: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        words = entry.get("words") or []
        text = " ".join((w.get("text") or "").strip() for w in words)
        text = " ".join(text.split())
        if not text:
            continue
        starts = [
            _ts(w.get("start_timestamp", w.get("start_time"))) for w in words
        ]
        ends = [_ts(w.get("end_timestamp", w.get("end_time"))) for w in words]
        starts = [s for s in starts if s is not None]
        ends = [e for e in ends if e is not None]
        if not starts:
            print(
                f"  WARNING: skipping transcript entry with no usable "
                f"timestamps: {json.dumps(entry)[:200]}",
                file=sys.stderr,
            )
            continue
        start = min(starts)
        end = max(ends) if ends else max(starts)
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": _speaker_label(entry),
                "text": text,
            }
        )
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


def write_outputs(segments: list[dict], outdir: str, basename: str) -> tuple[str, str]:
    """Write {basename}_segments.json and {basename}_transcript_speakers.txt.

    Line format matches cli.py exactly: "[MM:SS] {speaker}: {text}".
    """
    segments_path = os.path.join(outdir, f"{basename}_segments.json")
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)

    speakers_path = os.path.join(outdir, f"{basename}_transcript_speakers.txt")
    with open(speakers_path, "w") as f:
        for seg in segments:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            f.write(f"[{mins:02d}:{secs:02d}] {seg['speaker']}: {seg['text']}\n")
    return segments_path, speakers_path


# ---------------------------------------------------------------------------
# Probe summary
# ---------------------------------------------------------------------------

def print_probe_summary(bot: dict, raw, segments: list[dict]) -> None:
    word_counts: dict[str, int] = {}
    for seg in segments:
        word_counts[seg["speaker"]] = word_counts.get(seg["speaker"], 0) + len(
            seg["text"].split()
        )

    print("\n=== PROBE SUMMARY ===", flush=True)
    participants = bot.get("meeting_participants") or []
    if participants:
        names = [str(p.get("name") or f"id={p.get('id')}") for p in participants]
        print(f"participants (bot payload): {', '.join(names)}")
    else:
        print("participants (bot payload): none listed")
    event_count = len(raw) if isinstance(raw, list) else "n/a"
    print(f"transcript events: {event_count}")
    print(f"segments parsed:   {len(segments)}")
    print(f"speakers seen:     {len(word_counts)}")
    for speaker, count in sorted(word_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {speaker}: {count} words")
    if segments:
        print("first segments:")
        for seg in segments[:5]:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            preview = seg["text"][:80]
            print(f"  [{mins:02d}:{secs:02d}] {seg['speaker']}: {preview}")
    else:
        print("NO SEGMENTS PARSED — inspect raw_transcript.json before trusting Meet.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _basename_from_url(meeting_url: str) -> str:
    path = urllib.parse.urlparse(meeting_url).path
    tail = path.rstrip("/").split("/")[-1]
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in tail)
    return cleaned.strip("-") or "meeting"


def _dump(payload, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}", flush=True)
    return path


def main():
    _load_dotenv()
    _load_recall_key_fallback()

    parser = argparse.ArgumentParser(
        prog="meet-bot",
        description=(
            "Send a Recall.ai bot into a Google Meet call, wait for it to "
            "finish, and convert the transcript to meeting-debrief segments. "
            "Post-call only: no websockets, no webhooks, no tunnels."
        ),
    )
    parser.add_argument("meeting_url", help="Google Meet URL (https://meet.google.com/xxx-yyyy-zzz)")
    parser.add_argument("-o", "--output-dir", default="meet_capture", help="Output directory (default: ./meet_capture)")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=list(PROVIDERS),
        help=(
            "Transcription provider: recallai_async (Recall's built-in async "
            "transcription, $0.15/hr, works without captions; default) or "
            "meeting_captions (Meet's own captions; free, but the host must "
            "turn captions on)."
        ),
    )
    parser.add_argument("--probe", action="store_true", help="Dispatch, poll, dump raw JSONs, print a summary of what came back, and exit (go/no-go gate).")
    parser.add_argument("--timeout-min", type=int, default=90, help="Give up polling the bot after this many minutes (default: 90)")
    parser.add_argument("--basename", default=None, help="Basename for output files (default: derived from the meeting URL)")
    args = parser.parse_args()

    if not os.environ.get("RECALL_API_KEY"):
        print(
            "Error: RECALL_API_KEY not set (checked env, ./.env, repo-root "
            f".env, and {AIFUND_ENV_PATH}).",
            file=sys.stderr,
        )
        sys.exit(1)

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    basename = args.basename or _basename_from_url(args.meeting_url)

    print(f"Dispatching bot to {args.meeting_url} (provider={args.provider})...", flush=True)
    bot_id = dispatch(args.meeting_url, provider=args.provider)
    print(f"  bot id: {bot_id}", flush=True)
    print(
        f"Polling every {POLL_INTERVAL_S}s, timeout {args.timeout_min} min. "
        "If it sits in in_waiting_room, admit it from the meeting.",
        flush=True,
    )

    bot = poll(bot_id, timeout_min=args.timeout_min)
    _dump(bot, outdir, "raw_bot.json")

    changes = bot.get("status_changes") or []
    last = changes[-1] if changes else {}
    if last.get("code") == "fatal":
        print(
            f"Error: bot ended fatal "
            f"(sub_code={last.get('sub_code')}, message={last.get('message')}). "
            "Full payload: raw_bot.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Fetching transcript...", flush=True)
    raw = fetch_transcript(bot, provider=args.provider)
    _dump(raw, outdir, "raw_transcript.json")

    segments = transcript_to_segments(raw)

    if args.probe:
        print_probe_summary(bot, raw, segments)
        return

    if not segments:
        print(
            "Error: transcript produced no segments. Inspect "
            f"{os.path.join(outdir, 'raw_transcript.json')} — with "
            "meeting_captions this usually means captions were never enabled.",
            file=sys.stderr,
        )
        sys.exit(1)

    segments_path, speakers_path = write_outputs(segments, outdir, basename)
    print(f"  Saved: {segments_path}", flush=True)
    print(f"  Saved: {speakers_path}", flush=True)
    print(f"Done. {len(segments)} segments from {len({s['speaker'] for s in segments})} speakers.", flush=True)


if __name__ == "__main__":
    main()
