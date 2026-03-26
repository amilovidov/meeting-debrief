"""Local transcription using OpenAI Whisper."""

import os
import tempfile
import subprocess
from pathlib import Path


def transcribe(audio_path: str, model: str = "base", language: str = "en") -> list[dict]:
    """Transcribe audio using local Whisper model.

    Returns list of segments with start, end, text.
    """
    audio_path = str(Path(audio_path).resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "whisper", audio_path,
                "--model", model,
                "--language", language,
                "--output_format", "json",
                "--output_dir", tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Whisper failed: {result.stderr}")

        # Find the output JSON
        import json
        json_files = list(Path(tmpdir).glob("*.json"))
        if not json_files:
            raise RuntimeError("Whisper produced no output")

        with open(json_files[0]) as f:
            data = json.load(f)

        segments = []
        for seg in data.get("segments", []):
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
