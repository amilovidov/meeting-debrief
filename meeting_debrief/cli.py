"""CLI entry point for meeting-debrief."""

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="meeting-debrief",
        description="Fully local meeting/interview audio analysis with speaker diarization and vocal microexpressions.",
    )
    parser.add_argument("audio_file", help="Path to audio file (m4a, wav, mp3, etc.)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: same as audio file)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--language", default="en", help="Audio language (default: en)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Expected number of speakers (default: auto-detect)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote models")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--window", type=int, default=5, help="Analysis window size in minutes (default: 5)")
    parser.add_argument("--skip-diarization", action="store_true", help="Skip diarization (transcript-only analysis)")
    args = parser.parse_args()

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

    # Step 2: Transcribe
    print("\n[2/4] Transcribing with Whisper ({})...".format(args.whisper_model), flush=True)
    from meeting_debrief.transcribe import transcribe
    t0 = time.time()
    segments = transcribe(wav_path, model=args.whisper_model, language=args.language, device=args.device)
    print(f"  {len(segments)} segments in {time.time() - t0:.0f}s", flush=True)

    # Save transcript
    transcript_path = os.path.join(output_dir, f"{basename}_transcript.txt")
    with open(transcript_path, "w") as f:
        for seg in segments:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            f.write(f"[{mins:02d}:{secs:02d}] {seg['text']}\n")
    print(f"  Saved: {transcript_path}", flush=True)

    # Step 3: Diarize
    diarization = []
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
    else:
        print("\n[3/4] Skipping diarization", flush=True)

    # Step 3b: Merge transcript with diarization
    if diarization:
        print("\n  Merging transcript with speaker labels...", flush=True)
        from meeting_debrief.merge import merge_transcript_speakers
        merged = merge_transcript_speakers(segments, diarization)

        merged_path = os.path.join(output_dir, f"{basename}_transcript_speakers.txt")
        with open(merged_path, "w") as f:
            for seg in merged:
                mins = int(seg["start"] // 60)
                secs = int(seg["start"] % 60)
                f.write(f"[{mins:02d}:{secs:02d}] {seg['speaker']}: {seg['text']}\n")
        print(f"  Saved: {merged_path}", flush=True)

        # Also update segments with speaker info for analysis
        segments = merged

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
