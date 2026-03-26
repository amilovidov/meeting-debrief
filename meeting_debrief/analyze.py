"""Multi-layer conversation analysis inspired by the LDTMP engagement framework.

Layers:
1. Talk ratio per speaker per time window
2. Filler words (cognitive load)
3. Vocabulary diversity (engagement signal)
4. Conviction vs hedging keywords
5. Response quality (minimal/moderate/detailed)
6. Per-speaker pitch (F0) and expressiveness
7. Response latency between speakers
8. Turn duration patterns
9. Vocal energy per speaker (RMS)
10. Micro-pauses within turns (hesitation)
11. Engagement signals from shorter-speaking participant
"""

import re
from collections import Counter

import librosa
import numpy as np


# Word lists
STOP_WORDS = set(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could and but or nor for yet so if then than that "
    "this these those it its i me my we our you your he she they them their him "
    "her his what which who whom how when where why all each every some any no "
    "not very much more most also just only even still again already about into "
    "with from yeah right okay well going".split()
)

FILLER_WORDS = [
    "um", "uh", "you know", "i mean", "i guess",
    "essentially", "basically", "kind of", "sort of", "again",
]

EMOTIONAL_WORDS = set(
    "felt feeling happy sad excited nervous afraid love hate wonderful terrible "
    "amazing difficult joy sorrow peaceful anxious grateful disappointed proud "
    "worried confident uncertain frustrated passionate fascinated curious "
    "interesting engaged comfortable uncomfortable trust important missing "
    "personal meaningful cool insightful natural sticky profound".split()
)

HEDGING_WORDS = set(
    "maybe perhaps possibly somewhat probably potentially might could arguably".split()
)

CONVICTION_WORDS = set(
    "absolutely definitely clearly obviously certainly exactly precisely really "
    "truly actually important critical essential vital works worked pretty proud".split()
)


def analyze_transcript(
    segments: list[dict],
    diarization: list[dict],
    audio_path: str,
    window_minutes: int = 5,
) -> dict:
    """Run all analysis layers on transcript + diarization + audio.

    Returns dict with all analysis results.
    """
    window_size = window_minutes * 60
    duration = max(s["end"] for s in diarization) if diarization else 0
    max_window = int(duration // window_size) + 1

    # Identify speakers
    speaker_times = {}
    for seg in diarization:
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0) + seg["duration"]
    speakers = sorted(speaker_times.keys(), key=lambda s: speaker_times[s], reverse=True)
    primary = speakers[0] if speakers else "Speaker A"
    secondary = speakers[1] if len(speakers) > 1 else "Speaker B"

    # Load audio for acoustic analysis
    print("  Loading audio for acoustic analysis...", flush=True)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    results = {
        "speakers": {
            "primary": primary,
            "secondary": secondary,
            "primary_time": round(speaker_times.get(primary, 0) / 60, 1),
            "secondary_time": round(speaker_times.get(secondary, 0) / 60, 1),
        },
        "talk_ratio": _talk_ratio(diarization, primary, secondary, window_size, max_window),
        "fillers": _filler_analysis(segments, window_size),
        "vocabulary_diversity": _vocabulary_diversity(segments, window_size),
        "conviction_hedging": _conviction_hedging(segments, window_size),
        "response_quality": _response_quality(segments, window_size),
        "pitch": _pitch_analysis(diarization, y, sr, primary, secondary, window_size, max_window),
        "response_latency": _response_latency(diarization, primary, secondary, window_size, max_window),
        "turn_duration": _turn_duration(diarization, primary, secondary, window_size, max_window),
        "vocal_energy": _vocal_energy(diarization, y, sr, primary, secondary, window_size, max_window),
        "micro_pauses": _micro_pauses(diarization, y, sr, primary, window_size),
        "engagement_signals": _engagement_signals(diarization, secondary, window_size, max_window),
        "verbal_tics": _verbal_tics(segments),
        "answer_openers": _answer_openers(segments, diarization, primary),
    }

    return results


def _talk_ratio(diarization, primary, secondary, window_size, max_window):
    windows = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        p_time = sum(
            min(s["end"], w_end) - max(s["start"], w_start)
            for s in diarization if s["speaker"] == primary
            and s["end"] > w_start and s["start"] < w_end
        )
        s_time = sum(
            min(s["end"], w_end) - max(s["start"], w_start)
            for s in diarization if s["speaker"] == secondary
            and s["end"] > w_start and s["start"] < w_end
        )
        if p_time + s_time < 10:
            continue
        windows.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "primary_seconds": round(p_time),
            "secondary_seconds": round(s_time),
            "ratio": round(p_time / max(s_time, 1), 1),
        })
    return windows


def _filler_analysis(segments, window_size):
    filler_windows = {}
    for seg in segments:
        w = int(seg["start"] // window_size)
        if w not in filler_windows:
            filler_windows[w] = {f: 0 for f in FILLER_WORDS}
            filler_windows[w]["_words"] = 0
        text = seg["text"].lower()
        words = text.split()
        filler_windows[w]["_words"] += len(words)
        for filler in FILLER_WORDS:
            if " " in filler:
                filler_windows[w][filler] += text.count(filler)
            else:
                filler_windows[w][filler] += sum(
                    1 for word in words if word.strip(".,!?\"'()") == filler
                )

    results = []
    for w in sorted(filler_windows.keys()):
        d = filler_windows[w]
        total_f = sum(d[k] for k in d if not k.startswith("_"))
        tw = d["_words"]
        if tw < 30:
            continue
        results.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "total_fillers": total_f,
            "total_words": tw,
            "filler_pct": round(total_f / tw * 100, 1),
            "breakdown": {k: d[k] for k in FILLER_WORDS if d[k] > 0},
        })
    return results


def _vocabulary_diversity(segments, window_size):
    windows = {}
    for seg in segments:
        w = int(seg["start"] // window_size)
        if w not in windows:
            windows[w] = []
        words = [
            word.strip(".,!?\"'()").lower()
            for word in seg["text"].split()
            if word.strip(".,!?\"'()").lower() not in STOP_WORDS
            and len(word.strip(".,!?\"'()")) > 2
        ]
        windows[w].extend(words)

    results = []
    for w in sorted(windows.keys()):
        wl = windows[w]
        if len(wl) < 20:
            continue
        unique = len(set(wl))
        total = len(wl)
        results.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "diversity": round(unique / total, 3),
            "unique": unique,
            "total": total,
        })
    return results


def _conviction_hedging(segments, window_size):
    results = []
    windows = {}
    for seg in segments:
        w = int(seg["start"] // window_size)
        if w not in windows:
            windows[w] = {"emotional": 0, "hedging": 0, "conviction": 0, "total": 0}
        words = [word.strip(".,!?\"'()").lower() for word in seg["text"].split()]
        windows[w]["total"] += len(words)
        windows[w]["emotional"] += sum(1 for w2 in words if w2 in EMOTIONAL_WORDS)
        windows[w]["hedging"] += sum(1 for w2 in words if w2 in HEDGING_WORDS)
        windows[w]["conviction"] += sum(1 for w2 in words if w2 in CONVICTION_WORDS)

    for w in sorted(windows.keys()):
        d = windows[w]
        if d["total"] < 30:
            continue
        ratio = round(d["conviction"] / max(d["hedging"], 1), 1)
        results.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "emotional_pct": round(d["emotional"] / d["total"] * 100, 1),
            "hedging_pct": round(d["hedging"] / d["total"] * 100, 1),
            "conviction_pct": round(d["conviction"] / d["total"] * 100, 1),
            "ch_ratio": ratio,
        })
    return results


def _response_quality(segments, window_size):
    results = []
    windows = {}
    for seg in segments:
        w = int(seg["start"] // window_size)
        if w not in windows:
            windows[w] = {"minimal": 0, "moderate": 0, "detailed": 0}
        wc = len(seg["text"].split())
        if wc < 20:
            windows[w]["minimal"] += 1
        elif wc < 50:
            windows[w]["moderate"] += 1
        else:
            windows[w]["detailed"] += 1

    for w in sorted(windows.keys()):
        d = windows[w]
        total = d["minimal"] + d["moderate"] + d["detailed"]
        if total == 0:
            continue
        results.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            **d,
        })
    return results


def _pitch_analysis(diarization, y, sr, primary, secondary, window_size, max_window):
    print("  Computing per-speaker pitch...", flush=True)
    results = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        row = {"window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min"}

        for spk, label in [(primary, "primary"), (secondary, "secondary")]:
            samples = []
            for seg in diarization:
                if seg["speaker"] != spk:
                    continue
                seg_start = max(seg["start"], w_start)
                seg_end = min(seg["end"], w_end)
                if seg_end <= seg_start:
                    continue
                samples.append(y[int(seg_start * sr):int(seg_end * sr)])

            if not samples or sum(len(s) for s in samples) < sr:
                row[f"{label}_avg_hz"] = 0
                row[f"{label}_std"] = 0
                row[f"{label}_expressiveness"] = 0
                continue

            audio = np.concatenate(samples)
            f0, _, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=sr, frame_length=2048)
            voiced = f0[~np.isnan(f0)]

            if len(voiced) > 5:
                avg = float(np.mean(voiced))
                std = float(np.std(voiced))
                row[f"{label}_avg_hz"] = round(avg)
                row[f"{label}_std"] = round(std)
                row[f"{label}_expressiveness"] = round(std / max(avg, 1) * 100, 1)
            else:
                row[f"{label}_avg_hz"] = 0
                row[f"{label}_std"] = 0
                row[f"{label}_expressiveness"] = 0

        if row.get("primary_avg_hz", 0) > 0 or row.get("secondary_avg_hz", 0) > 0:
            results.append(row)

    return results


def _response_latency(diarization, primary, secondary, window_size, max_window):
    latencies = []
    for i in range(len(diarization) - 1):
        if diarization[i]["speaker"] == secondary and diarization[i + 1]["speaker"] == primary:
            latency = diarization[i + 1]["start"] - diarization[i]["end"]
            if 0 < latency < 10:
                latencies.append({
                    "latency": round(latency, 2),
                    "timestamp": diarization[i]["end"],
                })

    windows = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        wl = [l["latency"] for l in latencies if w_start <= l["timestamp"] < w_end]
        if not wl:
            continue
        windows.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "avg": round(float(np.mean(wl)), 2),
            "min": round(min(wl), 2),
            "max": round(max(wl), 2),
            "count": len(wl),
        })

    longest = sorted(latencies, key=lambda x: x["latency"], reverse=True)[:10]
    return {"windows": windows, "longest": longest}


def _turn_duration(diarization, primary, secondary, window_size, max_window):
    results = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        row = {"window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min"}
        for spk, label in [(primary, "primary"), (secondary, "secondary")]:
            turns = [s["duration"] for s in diarization
                     if s["speaker"] == spk and s["end"] > w_start and s["start"] < w_end]
            if turns:
                row[f"{label}_avg"] = round(float(np.mean(turns)), 1)
                row[f"{label}_max"] = round(max(turns), 1)
            else:
                row[f"{label}_avg"] = 0
                row[f"{label}_max"] = 0
        if row.get("primary_avg", 0) > 0 or row.get("secondary_avg", 0) > 0:
            results.append(row)
    return results


def _vocal_energy(diarization, y, sr, primary, secondary, window_size, max_window):
    print("  Computing per-speaker vocal energy...", flush=True)
    results = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        row = {"window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min"}
        for spk, label in [(primary, "primary"), (secondary, "secondary")]:
            samples = []
            for seg in diarization:
                if seg["speaker"] != spk:
                    continue
                seg_start = max(seg["start"], w_start)
                seg_end = min(seg["end"], w_end)
                if seg_end <= seg_start:
                    continue
                samples.append(y[int(seg_start * sr):int(seg_end * sr)])
            if samples:
                audio = np.concatenate(samples)
                row[f"{label}_energy"] = round(float(np.sqrt(np.mean(audio ** 2))), 4)
            else:
                row[f"{label}_energy"] = 0
        if row.get("primary_energy", 0) > 0 or row.get("secondary_energy", 0) > 0:
            results.append(row)
    return results


def _micro_pauses(diarization, y, sr, primary, window_size):
    print("  Detecting micro-pauses...", flush=True)
    results = {}
    for seg in diarization:
        if seg["speaker"] != primary or seg["duration"] < 2:
            continue
        w = int(seg["start"] // window_size)
        audio = y[int(seg["start"] * sr):int(seg["end"] * sr)]

        frame_len = int(0.05 * sr)
        energy = np.array([
            np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
            for i in range(0, len(audio) - frame_len, frame_len)
        ])
        threshold = np.mean(energy) * 0.15
        silent = energy < threshold

        pause_count = 0
        consecutive = 0
        for is_silent in silent:
            if is_silent:
                consecutive += 1
            else:
                if consecutive >= 6:  # >0.3s
                    pause_count += 1
                consecutive = 0

        if w not in results:
            results[w] = {"pauses": 0, "turns": 0, "total_duration": 0}
        results[w]["pauses"] += pause_count
        results[w]["turns"] += 1
        results[w]["total_duration"] += seg["duration"]

    return [
        {
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "pauses": d["pauses"],
            "turns": d["turns"],
            "rate_per_min": round(d["pauses"] / max(d["total_duration"] / 60, 0.1), 1),
        }
        for w, d in sorted(results.items())
    ]


def _engagement_signals(diarization, secondary, window_size, max_window):
    results = []
    for w in range(max_window):
        w_start, w_end = w * window_size, (w + 1) * window_size
        turns = [s["duration"] for s in diarization
                 if s["speaker"] == secondary and s["end"] > w_start and s["start"] < w_end]
        if not turns:
            continue
        short = sum(1 for t in turns if t < 2)
        medium = sum(1 for t in turns if 2 <= t < 5)
        long = sum(1 for t in turns if t >= 5)
        results.append({
            "window": f"{w * window_size // 60}-{(w + 1) * window_size // 60}min",
            "short": short,
            "medium": medium,
            "long": long,
        })
    return results


def _verbal_tics(segments):
    """Find the most repeated filler/tic words across the full transcript."""
    all_text = " ".join(s["text"].lower() for s in segments)
    tics = {}
    candidates = ["essentially", "basically", "you know", "i mean", "i guess",
                   "kind of", "sort of", "like", "right", "again", "actually"]
    for word in candidates:
        count = all_text.count(word)
        if count >= 5:
            tics[word] = count
    return dict(sorted(tics.items(), key=lambda x: x[1], reverse=True))


def _answer_openers(segments, diarization, primary):
    """Analyze how the primary speaker starts their responses."""
    openers = Counter()
    for seg in segments:
        if len(seg["text"].split()) < 15:
            continue
        first = seg["text"].split()[0].lower().strip(".,!?")
        if first == "so":
            openers["'So...'"] += 1
        elif first in ("i", "i'm"):
            next_word = seg["text"].split()[1].lower() if len(seg["text"].split()) > 1 else ""
            if next_word == "think":
                openers["'I think...'"] += 1
            elif next_word == "guess":
                openers["'I guess...'"] += 1
            else:
                openers["'I...'"] += 1
        elif first == "yeah":
            openers["'Yeah...'"] += 1
        elif first == "essentially":
            openers["'Essentially...'"] += 1
        elif first in ("it's", "its", "it"):
            openers["'It's...'"] += 1
        elif first == "and":
            openers["'And...'"] += 1
        else:
            openers["Other"] += 1

    return dict(openers.most_common())
