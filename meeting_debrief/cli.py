"""CLI entry point for meeting-debrief."""

import argparse
import json
import os
import sys
import time


def _load_dotenv():
    """Load KEY=VALUE pairs from .env into os.environ (without overriding already-set vars).

    Looks in the current working directory and the repo root (parent of this package),
    so API keys (OPENAI_API_KEY, HF_TOKEN, DEEPGRAM_API_KEY) work without a manual export.
    """
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def main():
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="meeting-debrief",
        description="Fully local meeting/interview audio analysis with speaker diarization and vocal microexpressions.",
    )
    parser.add_argument("audio_file", help="Path to audio file (m4a, wav, mp3, etc.)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: same as audio file)")
    parser.add_argument("--engine", default="openai", choices=["whisper", "openai", "deepgram"], help="Transcription engine: 'openai' (API, ~$0.006/min, default), 'whisper' (local, free), or 'deepgram' (API, ~$0.0043/min, native diarization). Default: openai")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size: tiny, base, small, medium, large-v3 (default: large-v3)")
    parser.add_argument("--openai-model", default="gpt-4o-transcribe", help="OpenAI STT model when --engine openai (default: gpt-4o-transcribe)")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--openai-concurrency", type=int, default=8, help="Parallel OpenAI API calls (default: 8)")
    parser.add_argument("--openai-prompt", default="", help="Domain vocabulary (names, products, terms) to bias OpenAI STT against proper-noun confabulation on isolated turns. e.g. --openai-prompt 'EkLine, Puneet Arora, Mentat, Confluence'")
    parser.add_argument("--deepgram-model", default="nova-3", help="Deepgram model when --engine deepgram (default: nova-3)")
    parser.add_argument("--deepgram-api-key", default=None, help="Deepgram API key (falls back to DEEPGRAM_API_KEY env var)")
    parser.add_argument("--language", default="en", help="Audio language (default: en)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Expected number of speakers (default: auto-detect)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote models")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--window", type=int, default=5, help="Analysis window size in minutes (default: 5)")
    parser.add_argument("--skip-diarization", action="store_true", help="Skip diarization (transcript-only analysis). Not compatible with --engine openai.")
    args = parser.parse_args()

    if args.engine == "openai" and args.skip_diarization:
        print("Error: --engine openai requires diarization (uses speaker turns as transcription chunks).", file=sys.stderr)
        sys.exit(1)

    if args.engine == "deepgram" and args.skip_diarization:
        print("Error: --engine deepgram does diarization in the same call; --skip-diarization is incompatible.", file=sys.stderr)
        sys.exit(1)

    audio_path = os.path.abspath(args.audio_file)
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    t_total = time.time()

    # Step 1: Convert to WAV
    print("\n[1/4] Converting audio...", flush=True)
    from meeting_debrief.transcribe import convert_to_wav
    wav_path = convert_to_wav(audio_path)
    print(f"  Converted to {wav_path}", flush=True)

    # Step 2 & 3: Transcribe + Diarize. Order depends on engine.
    diarization = []

    if args.engine == "deepgram":
        # Deepgram engine: single API call returns diarized, speaker-labeled utterances.
        print(f"\n[2/4] Transcribing + diarizing with Deepgram ({args.deepgram_model})...", flush=True)
        from meeting_debrief.transcribe import transcribe_deepgram
        t0 = time.time()
        segments, diarization = transcribe_deepgram(
            wav_path,
            model=args.deepgram_model,
            language=args.language,
            api_key=args.deepgram_api_key,
        )
        print(f"  {len(segments)} utterances + {len(diarization)} diarization segments in {time.time() - t0:.0f}s", flush=True)

        diar_path = os.path.join(output_dir, f"{basename}_diarization.json")
        with open(diar_path, "w") as f:
            json.dump(diarization, f, indent=2)
        print(f"  Saved: {diar_path}", flush=True)

        print("\n[3/4] Diarization included in step 2 (Deepgram native)", flush=True)
    elif args.engine == "openai":
        # OpenAI engine: diarize first, then transcribe per-turn.
        if args.skip_diarization:
            # Already rejected during arg validation, but keep guard.
            raise RuntimeError("--engine openai requires diarization")

        print("\n[2/4] Speaker diarization...", flush=True)
        from meeting_debrief.diarize import diarize
        t0 = time.time()
        diarization = diarize(
            wav_path,
            num_speakers=args.num_speakers,
            hf_token=args.hf_token,
            device=args.device,
        )
        print(f"  {len(diarization)} segments in {time.time() - t0:.0f}s", flush=True)

        diar_path = os.path.join(output_dir, f"{basename}_diarization.json")
        with open(diar_path, "w") as f:
            json.dump(diarization, f, indent=2)
        print(f"  Saved: {diar_path}", flush=True)

        print(f"\n[3/4] Transcribing with OpenAI ({args.openai_model})...", flush=True)
        from meeting_debrief.transcribe import transcribe_openai
        t0 = time.time()
        try:
            segments = transcribe_openai(
                wav_path,
                diarization,
                model=args.openai_model,
                language=args.language,
                api_key=args.openai_api_key,
                concurrency=args.openai_concurrency,
                prompt=args.openai_prompt,
            )
            print(f"  {len(segments)} turns transcribed in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            # Degradation floor: OpenAI is the default engine, but a dead
            # quota/billing/auth state must not yield an empty transcript and a
            # silent exit 0. Fall back to local Whisper (free, no API) loudly,
            # reusing the diarization we already computed above.
            print(f"\n  WARNING: OpenAI transcription failed: {e}", file=sys.stderr)
            print(
                f"  Falling back to local Whisper ({args.whisper_model}). "
                f"Fix the OpenAI billing/quota to use the API engine.",
                file=sys.stderr,
            )
            from meeting_debrief.transcribe import transcribe
            from meeting_debrief.merge import merge_transcript_speakers
            t0 = time.time()
            segments = transcribe(
                wav_path, model=args.whisper_model, language=args.language, device=args.device
            )
            segments = merge_transcript_speakers(segments, diarization)
            print(f"  {len(segments)} segments via Whisper fallback in {time.time() - t0:.0f}s", flush=True)
    else:
        # Whisper engine: transcribe first, then diarize, then merge.
        print("\n[2/4] Transcribing with Whisper ({})...".format(args.whisper_model), flush=True)
        from meeting_debrief.transcribe import transcribe
        t0 = time.time()
        segments = transcribe(wav_path, model=args.whisper_model, language=args.language, device=args.device)
        print(f"  {len(segments)} segments in {time.time() - t0:.0f}s", flush=True)

        if not args.skip_diarization:
            print("\n[3/4] Speaker diarization...", flush=True)
            from meeting_debrief.diarize import diarize
            t0 = time.time()
            diarization = diarize(
                wav_path,
                num_speakers=args.num_speakers,
                hf_token=args.hf_token,
                device=args.device,
            )
            print(f"  {len(diarization)} segments in {time.time() - t0:.0f}s", flush=True)

            diar_path = os.path.join(output_dir, f"{basename}_diarization.json")
            with open(diar_path, "w") as f:
                json.dump(diarization, f, indent=2)
            print(f"  Saved: {diar_path}", flush=True)

            print("\n  Merging transcript with speaker labels...", flush=True)
            from meeting_debrief.merge import merge_transcript_speakers
            segments = merge_transcript_speakers(segments, diarization)
        else:
            print("\n[3/4] Skipping diarization", flush=True)

    # Backstop: never write an empty transcript and run a meaningless analysis
    # on it. If we got here with no segments, transcription failed entirely.
    if not segments:
        print(
            "\nError: transcription produced no text. Aborting before an empty "
            "analysis. Check the engine output above (API quota/billing/auth).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Save plain transcript (always)
    transcript_path = os.path.join(output_dir, f"{basename}_transcript.txt")
    with open(transcript_path, "w") as f:
        for seg in segments:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            f.write(f"[{mins:02d}:{secs:02d}] {seg['text']}\n")
    print(f"  Saved: {transcript_path}", flush=True)

    # Save speaker-labeled transcript when speakers are available
    if diarization and segments and "speaker" in segments[0]:
        merged_path = os.path.join(output_dir, f"{basename}_transcript_speakers.txt")
        with open(merged_path, "w") as f:
            for seg in segments:
                mins = int(seg["start"] // 60)
                secs = int(seg["start"] % 60)
                f.write(f"[{mins:02d}:{secs:02d}] {seg['speaker']}: {seg['text']}\n")
        print(f"  Saved: {merged_path}", flush=True)

    # Step 4: Analyze
    print("\n[4/4] Analyzing...", flush=True)
    from meeting_debrief.analyze import analyze_transcript
    t0 = time.time()
    results = analyze_transcript(
        segments=segments,
        diarization=diarization,
        audio_path=wav_path,
        window_minutes=args.window,
    )
    print(f"  Analysis complete in {time.time() - t0:.0f}s", flush=True)

    # Save raw results
    # Convert numpy types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        return obj

    raw_path = os.path.join(output_dir, f"{basename}_analysis.json")
    with open(raw_path, "w") as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"  Saved: {raw_path}", flush=True)

    # Generate report
    from meeting_debrief.report import generate_report
    report_path = os.path.join(output_dir, f"{basename}_report.md")
    generate_report(results, report_path)
    print(f"  Saved: {report_path}", flush=True)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.0f}s. Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
