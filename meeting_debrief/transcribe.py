"""Transcription engines: local Whisper or OpenAI API (gpt-4o-transcribe)."""

import os
import sys
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _select_device(preferred: str = "auto") -> str:
    """Select compute device: mps > cuda > cpu."""
    import torch
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transcribe(audio_path: str, model: str = "base", language: str = "en", device: str = "auto") -> list[dict]:
    """Transcribe audio using local Whisper model via Python API.

    Uses MPS on Apple Silicon, CUDA on NVIDIA, CPU as fallback.
    Returns list of segments with start, end, text.
    """
    import whisper

    device = _select_device(device)
    print(f"  Whisper device: {device}", file=sys.stderr)

    whisper_model = whisper.load_model(model, device=device)
    result = whisper_model.transcribe(
        str(Path(audio_path).resolve()),
        language=language,
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    return segments


def convert_to_wav(audio_path: str, output_path: str = None, sr: int = 16000) -> str:
    """Convert any audio format to 16kHz mono WAV for processing."""
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "meeting_debrief_audio.wav")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-map", "0:a:0",
            "-ac", "1", "-ar", str(sr),
            "-c:a", "pcm_s16le",
            output_path,
        ],
        capture_output=True,
    )

    if not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg conversion failed for {audio_path}")

    return output_path


def _group_diarization_into_turns(
    diarization: list[dict],
    gap_threshold: float = 1.0,
) -> list[dict]:
    """Group consecutive same-speaker diarization segments into turns.

    Merges segments from the same speaker when separated by less than
    `gap_threshold` seconds of silence.
    """
    turns: list[dict] = []
    for seg in diarization:
        if (
            turns
            and turns[-1]["speaker"] == seg["speaker"]
            and seg["start"] - turns[-1]["end"] < gap_threshold
        ):
            turns[-1]["end"] = seg["end"]
        else:
            turns.append(
                {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]}
            )
    return turns


def _transcribe_chunk_openai(
    client,
    audio_path: str,
    start: float,
    end: float,
    model: str,
    language: str,
    tmpdir: str,
    idx: int,
    prompt: str = "",
) -> str:
    """Extract a chunk of audio and transcribe it via the OpenAI API.

    Each turn is transcribed in isolation, so gpt-4o-transcribe has no
    conversational context and will confabulate plausible-but-wrong tokens
    on short or boundary-clipped chunks (e.g. inventing "MIT" for an unclear
    word). `prompt` biases the model toward the meeting's real vocabulary —
    proper nouns, company/product names, domain terms — which suppresses that
    failure. It is a soft bias, not a hard constraint.
    """
    chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.mp3")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-c:a", "libmp3lame", "-b:a", "64k",
            chunk_path,
        ],
        capture_output=True,
        check=True,
    )
    create_kwargs = {
        "model": model,
        "language": language,
        "response_format": "text",
    }
    if prompt:
        create_kwargs["prompt"] = prompt
    with open(chunk_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, **create_kwargs)
    text = response if isinstance(response, str) else response.text
    try:
        os.remove(chunk_path)
    except OSError:
        pass
    return text.strip()


def transcribe_openai(
    audio_path: str,
    diarization: list[dict],
    model: str = "gpt-4o-transcribe",
    language: str = "en",
    api_key: str | None = None,
    concurrency: int = 8,
    min_turn_duration: float = 0.8,
    prompt: str = "",
) -> list[dict]:
    """Transcribe audio per-turn using the OpenAI API (gpt-4o-transcribe).

    Uses diarization to split the audio into per-speaker turns and transcribes
    each turn independently. Returns segments natively labeled with speakers,
    so no separate merge step is needed.

    Args:
        audio_path: 16kHz mono WAV from convert_to_wav.
        diarization: pyannote output — list of {start, end, speaker, duration}.
        model: OpenAI STT model. Default gpt-4o-transcribe (highest quality).
        language: ISO-639-1 code.
        api_key: OPENAI_API_KEY (falls back to env).
        concurrency: parallel API calls.
        min_turn_duration: skip turns shorter than this (backchannels).
        prompt: domain vocabulary (names, products, terms) to bias each
            isolated-chunk transcription against proper-noun confabulation.

    Returns:
        List of {start, end, speaker, text} sorted by start time.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Install with: pip install openai"
        ) from e

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it or pass --openai-api-key."
        )
    client = OpenAI(api_key=key)

    turns = _group_diarization_into_turns(diarization)
    turns = [t for t in turns if t["end"] - t["start"] >= min_turn_duration]
    print(
        f"  OpenAI engine: {len(turns)} turns to transcribe "
        f"(model={model}, concurrency={concurrency})",
        file=sys.stderr,
    )

    results: list[dict | None] = [None] * len(turns)
    completed = 0
    failures = 0
    last_err: Exception | None = None

    with tempfile.TemporaryDirectory(prefix="meeting_debrief_oai_") as tmpdir:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_idx = {
                ex.submit(
                    _transcribe_chunk_openai,
                    client, audio_path,
                    t["start"], t["end"],
                    model, language, tmpdir, i, prompt,
                ): i
                for i, t in enumerate(turns)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                t = turns[idx]
                try:
                    text = fut.result()
                except Exception as e:
                    print(f"  Turn {idx} ({t['start']:.1f}s) failed: {e}", file=sys.stderr)
                    text = ""
                    failures += 1
                    last_err = e
                results[idx] = {
                    "start": round(t["start"], 2),
                    "end": round(t["end"], 2),
                    "speaker": t["speaker"],
                    "text": text,
                }
                completed += 1
                if completed % 10 == 0 or completed == len(turns):
                    print(f"  {completed}/{len(turns)} turns done", file=sys.stderr)

    non_empty = [r for r in results if r and r["text"]]
    # Fail loudly when every turn failed. Returning an empty list here silently
    # produces an empty transcript and a meaningless analysis run that still
    # exits 0 (the insufficient_quota case). The caller catches this to fall
    # back to local Whisper.
    if turns and not non_empty:
        raise RuntimeError(
            f"OpenAI transcription failed on all {len(turns)} turns "
            f"(0 produced text). Check OPENAI_API_KEY quota/billing/auth. "
            f"Last error: {last_err}"
        )
    return non_empty


def transcribe_deepgram(
    audio_path: str,
    model: str = "nova-3",
    language: str = "en",
    api_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Transcribe + diarize audio in one call using the Deepgram API (nova-3).

    Deepgram does speaker diarization natively, so pyannote is skipped.
    Returns both the speaker-labeled segments AND a diarization timeline
    derived from utterances, so the audio-analysis layer still gets
    per-speaker timings.

    Args:
        audio_path: any audio file ffmpeg can read (we send raw bytes).
        model: Deepgram model. nova-3 / nova-3-general / nova-3-medical / nova-2-* / etc.
        language: ISO-639-1 code.
        api_key: DEEPGRAM_API_KEY (falls back to env).

    Returns:
        (segments, diarization)
        segments: [{start, end, speaker, text}] sorted by start.
        diarization: [{start, end, speaker}] same shape pyannote produces.
        Speaker labels are normalized to pyannote-style "SPEAKER_00", "SPEAKER_01".
    """
    try:
        from deepgram import DeepgramClient
    except ImportError as e:
        raise RuntimeError(
            "deepgram-sdk not installed. Install with: pip install deepgram-sdk"
        ) from e

    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY not set. Export it or pass --deepgram-api-key."
        )

    client = DeepgramClient(api_key=key)

    print(f"  Deepgram engine: model={model}, diarize=true, utterances=true", file=sys.stderr)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model=model,
        language=language,
        diarize=True,
        utterances=True,
        smart_format=True,
        punctuate=True,
        paragraphs=True,
        filler_words=True,
    )

    # Response is a typed object; .results.utterances has what we need.
    utterances = response.results.utterances or []

    segments: list[dict] = []
    diarization: list[dict] = []
    for utt in utterances:
        speaker_label = f"SPEAKER_{int(utt.speaker):02d}"
        text = (utt.transcript or "").strip()
        if not text:
            continue
        segments.append({
            "start": round(float(utt.start), 2),
            "end": round(float(utt.end), 2),
            "speaker": speaker_label,
            "text": text,
        })
        diarization.append({
            "start": round(float(utt.start), 2),
            "end": round(float(utt.end), 2),
            "duration": round(float(utt.end) - float(utt.start), 2),
            "speaker": speaker_label,
        })

    print(f"  Deepgram returned {len(segments)} utterances", file=sys.stderr)
    return segments, diarization
