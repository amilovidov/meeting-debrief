"""Extract feature requests and design discussions from a meeting transcript.

This is the package's first generative stage. Everything else in
meeting-debrief is deterministic (STT + diarization + metric analysis);
this module takes the speaker-attributed segments and asks an LLM to
aggregate the discussion into structured feature-request objects that a
backlog (GitHub issues) or an MCP write tool can consume.

Design rules:
  - Every extracted item must be anchored by verbatim quotes with
    timestamps. Empty lists over invented content: if acceptance
    criteria were never discussed, the list stays empty.
  - Degradation floor: if the LLM call fails for any reason, we still
    write a useful artifact - the speaker-attributed transcript grouped
    into discussion blocks, honestly labeled as unprocessed. Exit code 2
    signals the degraded path; the pipeline downstream can still deliver
    the raw capture.

Input: either a segments JSON (list of {start, end, speaker, text}) or a
"*_transcript_speakers.txt" file in the package's own line format
("[MM:SS] Speaker A: text").
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

# JSON schema for OpenAI structured outputs (strict mode: every object
# closed with additionalProperties=false, all fields required).
_QUOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "ts": {"type": "string", "description": "MM:SS timestamp of the quote"},
        "speaker": {"type": "string"},
        "text": {"type": "string", "description": "verbatim quote from the transcript"},
    },
    "required": ["ts", "speaker", "text"],
    "additionalProperties": False,
}

FEATURE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "short imperative title"},
                    "kind": {
                        "type": "string",
                        "enum": ["feature", "bug", "design-decision", "integration", "process"],
                    },
                    "problem": {
                        "type": "string",
                        "description": "the user problem or need, as actually discussed",
                    },
                    "discussed_detail": {
                        "type": "string",
                        "description": (
                            "aggregation of EVERYTHING said about this item across the whole "
                            "meeting, in markdown; preserve concrete specifics (names, URLs, "
                            "constraints, numbers) exactly as spoken"
                        ),
                    },
                    "proposer": {"type": "string", "description": "speaker who first raised it"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "decisions": {"type": "array", "items": {"type": "string"}},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ONLY criteria stated or directly implied in discussion",
                    },
                    "quotes": {"type": "array", "items": _QUOTE_SCHEMA},
                },
                "required": [
                    "title", "kind", "problem", "discussed_detail", "proposer",
                    "participants", "decisions", "open_questions",
                    "acceptance_criteria", "quotes",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["feature_requests"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You extract feature requests and design discussions from an engineering \
meeting transcript with speaker attribution.

Rules, in priority order:
1. Only include items that were actually discussed in the transcript. \
Never invent, embellish, or extrapolate beyond what was said.
2. Every item MUST include verbatim supporting quotes: each quote is a \
single CONTIGUOUS span copied character-for-character from one \
transcript line, with its timestamp and speaker. Never stitch words \
from different places with ellipses; if a line is long, quote the \
relevant contiguous part of it. An item you cannot quote is an item \
you must not emit.
3. Aggregate, but do not over-collapse. If the same feature comes up at \
05:10, 12:40 and 21:05, that is ONE item whose discussed_detail weaves \
together all three passes. But DISTINCT capabilities, bugs, decisions \
and integration mechanisms are SEPARATE items even when they serve one \
larger goal: a half-hour design discussion typically yields 3-10 items, \
not 1. If you find yourself writing one item that contains several \
independently buildable things, split it.
4. discussed_detail carries the full richness of the conversation: \
constraints, alternatives considered, who pushed back and why, concrete \
names/numbers/URLs mentioned. It should let an engineer who missed the \
meeting build the right thing.
5. Empty lists over invention: if no acceptance criteria were stated or \
directly implied, leave acceptance_criteria empty. Same for decisions \
and open_questions.
6. kind: "feature" for new capability requests, "bug" for defects, \
"design-decision" for agreed architecture/approach, "integration" for \
cross-product wiring, "process" for how-we-work items.
7. Ignore small talk, scheduling, and pleasantries entirely.
"""

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["feature", "bug", "design-decision", "integration", "process"],
                    },
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "MM:SS timestamps where this item is discussed",
                    },
                },
                "required": ["title", "kind", "timestamps"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

_SWEEP_PROMPT = """\
You are doing a COVERAGE SWEEP of an engineering meeting transcript. \
Walk it start to finish and list EVERY distinct item of these kinds:
- feature: a capability someone wants built
- bug: a defect or error someone reports (even in passing)
- design-decision: an approach/architecture the participants settled on
- integration: cross-product wiring discussed
- process: how-we-work agreements (testbeds, pilots, syncs)

Include items mentioned only briefly. Err on the side of listing more, \
smaller, independently buildable items rather than one umbrella item. \
A stated future vision or desired end-state workflow ("we should be \
able to...", "where I'm going is...") IS a feature item - capture it \
with kind "feature", however far off it sounds. Only exclude small \
talk and scheduling chatter. For each item give the MM:SS timestamps \
where it comes up (multiple if revisited).
"""

_LINE_RE = re.compile(r"^\[(\d{1,3}):(\d{2})\]\s+([^:]+):\s+(.*)$")


def parse_speaker_transcript(path: str) -> list[dict]:
    """Parse the package's "[MM:SS] Speaker: text" format into segments."""
    segments = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = _LINE_RE.match(line)
            if not m:
                # Loud, not silent: a malformed line means the wrong file
                # was passed or the format drifted.
                raise ValueError(
                    f"{path}:{lineno}: line does not match '[MM:SS] Speaker: text': {line!r}"
                )
            mins, secs, speaker, text = m.groups()
            start = int(mins) * 60 + int(secs)
            segments.append(
                {"start": float(start), "end": float(start), "speaker": speaker.strip(), "text": text}
            )
    if not segments:
        raise ValueError(f"{path}: no transcript lines parsed")
    return segments


def load_segments(path: str) -> list[dict]:
    """Load segments from a .json segments file or a *_transcript_speakers.txt."""
    if path.endswith(".json"):
        with open(path) as f:
            segments = json.load(f)
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{path}: expected a non-empty JSON list of segments")
        for i, seg in enumerate(segments):
            missing = {"start", "speaker", "text"} - set(seg)
            if missing:
                raise ValueError(f"{path}: segment {i} missing keys {sorted(missing)}")
        return segments
    return parse_speaker_transcript(path)


def apply_speaker_map(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Replace anonymous speaker labels with real names ('Speaker A' -> 'Karthik')."""
    if not mapping:
        return segments
    return [
        {**seg, "speaker": mapping.get(seg["speaker"], seg["speaker"])}
        for seg in segments
    ]


def parse_speaker_map_arg(arg: str) -> dict[str, str]:
    """Parse 'Speaker A=Karthik,Speaker B=Alex' into a dict."""
    mapping = {}
    for pair in arg.split(","):
        if "=" not in pair:
            raise ValueError(f"--speaker-map entry {pair!r} is not 'Label=Name'")
        label, name = pair.split("=", 1)
        mapping[label.strip()] = name.strip()
    return mapping


def _format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


# A 30-min call is ~20 KB of transcript; anything over this cap is not a
# meeting transcript and would need chunking this v1 does not do.
_MAX_TRANSCRIPT_BYTES = 200_000


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s.lower())


def verify_quotes(feature_requests: list[dict], segments: list[dict]) -> tuple[int, int]:
    """Code-enforced verbatim gate: drop any quote whose normalized text is
    not a substring of the normalized transcript.

    The LLM is INSTRUCTED to quote verbatim, but prompts are ~80% reliable;
    every quote that ships must be provably in the transcript. Mutates
    feature_requests in place. Returns (kept, dropped).
    """
    transcript = _norm(" ".join(seg["text"] for seg in segments))
    kept = dropped = 0
    for fr in feature_requests:
        good = []
        for q in fr.get("quotes", []):
            if _norm(q.get("text", "")) and _norm(q["text"]) in transcript:
                good.append(q)
                kept += 1
            else:
                dropped += 1
        fr["quotes"] = good
    return kept, dropped


def unaccounted_candidates(candidates: list[dict], feature_requests: list[dict]) -> list[dict]:
    """Candidates none of whose sweep timestamps appear in any final item's quotes.

    Deterministic observability for the detail pass's known failure mode
    (silent candidate drops): imperfect as a matcher, but a dropped
    discussion block reliably shows up here. Report, don't gate.
    """
    quoted_ts = {
        q.get("ts") for fr in feature_requests for q in fr.get("quotes", [])
    }
    return [
        c for c in candidates
        if not any(ts in quoted_ts for ts in c.get("timestamps", []))
    ]


def _structured_call(client, model: str, system: str, user: str, schema_name: str, schema: dict) -> dict:
    """One structured-outputs chat call; returns the parsed JSON object."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    )
    return json.loads(response.choices[0].message.content)


def extract_feature_requests(
    segments: list[dict],
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> dict:
    """Two-pass extraction: coverage sweep (recall), then detail pass (precision).

    Single-pass extraction measured at 1-4 items on a 28-min fixture whose
    ground truth is ~7; the sweep pass exists to fix that recall ceiling.
    Raises on any API failure (caller owns the degradation floor).
    """
    transcript = _format_transcript(segments)
    if len(transcript.encode()) > _MAX_TRANSCRIPT_BYTES:
        raise ValueError(
            f"transcript is {len(transcript.encode())} bytes (cap {_MAX_TRANSCRIPT_BYTES}); "
            "chunking is not implemented in v1 - split the input"
        )
    speakers = sorted({seg["speaker"] for seg in segments})

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Install with: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Export it or pass --openai-api-key.")

    client = OpenAI(api_key=key)
    user_msg = f"Speakers: {', '.join(speakers)}\n\nTranscript:\n\n{transcript}"

    # Pass 1: coverage sweep - enumerate every candidate item.
    sweep = _structured_call(
        client, model, _SWEEP_PROMPT, user_msg, "candidates", _CANDIDATE_SCHEMA
    )
    candidates = sweep.get("candidates", [])
    if not candidates:
        raise RuntimeError("coverage sweep returned zero candidates")

    # Pass 2: detail each candidate against the full transcript.
    candidate_list = "\n".join(
        f"- [{c['kind']}] {c['title']} (discussed at {', '.join(c['timestamps'])})"
        for c in candidates
    )
    detail_user = (
        f"{user_msg}\n\n---\n\n"
        "A coverage sweep identified these candidate items:\n\n"
        f"{candidate_list}\n\n"
        "Produce the final feature-request list from these candidates. "
        "EVERY candidate must be accounted for: either it becomes its own "
        "item, or it is merged into another item because it is the SAME "
        "buildable thing (then its content appears in that item's "
        "discussed_detail). Never silently drop a candidate; the only valid "
        "reason to omit one is that the transcript gives it no quotable "
        "support at its timestamps. Expect the final count to be close to "
        "the candidate count."
    )
    result = _structured_call(
        client, model, _SYSTEM_PROMPT, detail_user, "feature_requests", FEATURE_REQUEST_SCHEMA
    )
    if "feature_requests" not in result:
        raise RuntimeError(f"model returned unexpected shape: {list(result)}")
    # Keep the sweep output for recall debugging: a missed item is either
    # a pass-1 miss (not in candidates) or a pass-2 drop (in candidates,
    # absent from the final list).
    result["candidates"] = candidates
    return result


def render_markdown(feature_requests: list[dict], source: str) -> str:
    """Human-review rendering of the extracted items."""
    today = datetime.date.today().isoformat()
    out = [f"# Feature requests extracted from meeting\n"]
    out.append(f"Source: `{source}` · extracted {today} · {len(feature_requests)} item(s)\n")
    for i, fr in enumerate(feature_requests, 1):
        out.append(f"\n## {i}. {fr['title']}  `[{fr['kind']}]`\n")
        out.append(f"**Proposer:** {fr['proposer']} · **Participants:** {', '.join(fr['participants']) or '-'}\n")
        out.append(f"\n### Problem\n\n{fr['problem']}\n")
        out.append(f"\n### Discussed detail\n\n{fr['discussed_detail']}\n")
        for key, heading in (
            ("decisions", "Decisions"),
            ("open_questions", "Open questions"),
            ("acceptance_criteria", "Acceptance criteria"),
        ):
            if fr[key]:
                out.append(f"\n### {heading}\n")
                out.extend(f"- {item}" for item in fr[key])
                out.append("")
        if fr["quotes"]:
            out.append("\n### Source quotes\n")
            out.extend(
                f"> [{q['ts']}] {q['speaker']}: {q['text']}\n" for q in fr["quotes"]
            )
    return "\n".join(out) + "\n"


# Gap (seconds) between consecutive segments that starts a new discussion
# block in the degraded output. 45s of silence/unattributed time in a
# meeting reliably marks a topic boundary; within-topic pauses are shorter.
_BLOCK_GAP_S = 45.0


def degradation_floor(segments: list[dict], error: Exception, out_base: str) -> str:
    """LLM failed: write the honest-minimal artifact (grouped raw capture).

    Returns the path written. Never raises: this is the floor.
    """
    path = f"{out_base}_feature_requests_raw.md"
    blocks: list[list[dict]] = [[]]
    prev_start = None
    for seg in segments:
        if prev_start is not None and seg["start"] - prev_start > _BLOCK_GAP_S:
            blocks.append([])
        blocks[-1].append(seg)
        prev_start = seg["start"]

    lines = [
        "# Meeting capture (UNPROCESSED)\n",
        "Feature-request extraction FAILED; this is the raw speaker-attributed "
        "capture grouped into discussion blocks. Re-run extraction when the "
        "API is available.\n",
        f"Extraction error: `{type(error).__name__}: {error}`\n",
    ]
    for i, block in enumerate(b for b in blocks if b):
        lines.append(f"\n## Discussion block {i + 1}\n")
        for seg in block:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {seg['speaker']}: {seg['text']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract feature requests from a speaker-attributed meeting transcript"
    )
    parser.add_argument("input", help="segments .json or *_transcript_speakers.txt")
    parser.add_argument("-o", "--output-dir", default=None, help="default: alongside input")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument(
        "--speaker-map", default=None,
        help="rename speakers, e.g. 'Speaker A=Karthik,Speaker B=Alex'",
    )
    args = parser.parse_args()

    # Reuse the package's .env loading so OPENAI_API_KEY resolves the same
    # way it does for transcription.
    from meeting_debrief.cli import _load_dotenv
    _load_dotenv()

    segments = load_segments(args.input)
    if args.speaker_map:
        segments = apply_speaker_map(segments, parse_speaker_map_arg(args.speaker_map))

    base = os.path.basename(args.input)
    for suffix in ("_transcript_speakers.txt", "_segments.json", ".json", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, base)

    print(f"Extracting feature requests from {len(segments)} segments "
          f"({len(_format_transcript(segments))} chars) via {args.model}...", flush=True)
    try:
        result = extract_feature_requests(
            segments, model=args.model, api_key=args.openai_api_key
        )
    except Exception as e:  # noqa: BLE001 - the floor catches everything
        print(f"\nExtraction FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        path = degradation_floor(segments, e, out_base)
        print(f"Degraded artifact written: {path}", file=sys.stderr)
        sys.exit(2)

    frs = result["feature_requests"]
    kept, dropped_q = verify_quotes(frs, segments)
    if dropped_q:
        print(f"  WARNING: {dropped_q} quote(s) failed the verbatim check and were dropped "
              f"({kept} kept)", file=sys.stderr)
        for fr in frs:
            if not fr["quotes"]:
                print(f"  WARNING: item has no surviving quotes: {fr['title']!r}",
                      file=sys.stderr)
    missing = unaccounted_candidates(result.get("candidates", []), frs)
    if missing:
        print(f"  WARNING: {len(missing)} sweep candidate(s) not traceable to any "
              "final item's quotes (possible detail-pass drop; consider re-running):",
              file=sys.stderr)
        for c in missing:
            print(f"    - [{c['kind']}] {c['title']} @ {', '.join(c['timestamps'])}",
                  file=sys.stderr)
    with open(f"{out_base}_candidates.json", "w") as f:
        json.dump(result.get("candidates", []), f, indent=2)
    json_path = f"{out_base}_feature_requests.json"
    with open(json_path, "w") as f:
        json.dump(frs, f, indent=2)
    md_path = f"{out_base}_feature_requests.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(frs, args.input))
    print(f"  {len(frs)} feature request(s)")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
