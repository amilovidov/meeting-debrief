"""Merge Whisper transcript segments with pyannote speaker diarization."""


def merge_transcript_speakers(
    transcript: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcript segments based on diarization.

    For each transcript segment, finds the diarization segment with the
    most overlap and assigns that speaker label.

    Args:
        transcript: List of dicts with start, end, text.
        diarization: List of dicts with start, end, speaker.

    Returns:
        List of dicts with start, end, text, speaker.
    """
    merged = []
    for seg in transcript:
        seg_start = seg["start"]
        seg_end = seg.get("end", seg_start + 5)

        # Find diarization segment with most overlap
        best_speaker = "Unknown"
        best_overlap = 0

        for diar in diarization:
            overlap_start = max(seg_start, diar["start"])
            overlap_end = min(seg_end, diar["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]

        merged.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
            "speaker": best_speaker,
        })

    return merged
