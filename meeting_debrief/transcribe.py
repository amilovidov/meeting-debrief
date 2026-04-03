"""Local transcription using OpenAI Whisper."""

import os
import sys
import tempfile
import subprocess
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
