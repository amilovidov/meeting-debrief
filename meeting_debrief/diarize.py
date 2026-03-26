"""Speaker diarization using pyannote.audio.

Supports Apple Silicon MPS via the gpu-optimizations fork.
Falls back to CPU if MPS is unavailable.
"""

import json
import os
import time
import warnings

import torch
import torchaudio


def diarize(
    audio_path: str,
    num_speakers: int = None,
    hf_token: str = None,
    device: str = "auto",
) -> list[dict]:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to WAV file (16kHz mono recommended).
        num_speakers: Expected number of speakers (None for auto-detect).
        hf_token: HuggingFace token for model access.
        device: "auto" (detect MPS/CUDA/CPU), "mps", "cuda", or "cpu".

    Returns:
        List of dicts with start, end, duration, speaker keys.
    """
    warnings.filterwarnings("ignore")

    # Resolve HF token
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf-token.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Then accept model access at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)

    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    torch_device = torch.device(device)
    print(f"  Device: {device}", flush=True)

    # Load pipeline
    from pyannote.audio import Pipeline

    print("  Loading diarization model...", flush=True)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if device != "cpu":
        pipeline = pipeline.to(torch_device)

    # Load audio into memory for faster processing
    print("  Loading audio into memory...", flush=True)
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sample_rate

    # Run diarization with progress hook
    print(f"  Running diarization ({duration/60:.1f} min of audio)...", flush=True)
    t0 = time.time()

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            result = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                **kwargs,
            )
    except Exception:
        # Fallback without progress hook
        result = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **kwargs,
        )

    elapsed = time.time() - t0
    print(f"  Diarization complete in {elapsed:.0f}s ({elapsed/duration:.2f}x realtime)", flush=True)

    # Extract segments — handle both old and new pyannote output formats
    segments = []
    annotation = None

    if hasattr(result, "itertracks"):
        annotation = result
    elif hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
    elif hasattr(result, "annotation"):
        annotation = result.annotation

    if annotation is not None:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "duration": round(turn.end - turn.start, 2),
                "speaker": speaker,
            })
    else:
        raise RuntimeError(f"Unexpected diarization output type: {type(result)}")

    # Map speaker IDs to readable names
    speaker_times = {}
    for seg in segments:
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0) + seg["duration"]

    # Sort by total speaking time descending
    ranked = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
    speaker_map = {spk: f"Speaker {chr(65 + i)}" for i, (spk, _) in enumerate(ranked)}

    for seg in segments:
        seg["speaker"] = speaker_map[seg["speaker"]]

    print(f"  Detected {len(speaker_map)} speakers:", flush=True)
    for old_id, new_name in speaker_map.items():
        t = speaker_times[old_id]
        print(f"    {new_name}: {t/60:.1f} min", flush=True)

    return segments
