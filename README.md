# meeting-debrief

Fully local meeting and interview audio analysis. One command: audio file in, detailed analysis report out. Zero API calls, zero cost.

Speaker diarization, vocal microexpressions, engagement scoring, and conversation dynamics — all running on your machine.

## What it does

Drop in any audio recording of a meeting or interview. Get back:

- **Speaker-separated transcript** with timestamps
- **Talk ratio** per speaker over time
- **Filler word tracking** (cognitive load signal)
- **Vocabulary diversity** (engagement signal — drops before disengagement)
- **Conviction vs hedging ratio** (confidence signal)
- **Per-speaker pitch analysis** (excitement, stress, expressiveness)
- **Response latency** (hesitation moments between speakers)
- **Turn duration patterns** (monologue vs dialogue detection)
- **Vocal energy** per speaker (who's louder, when)
- **Micro-pause detection** within turns (mid-thought hesitation)
- **Engagement signals** from the quieter speaker
- **Verbal tic detection** (repeated filler words)
- **Answer opener patterns** (how you start responses)

## Background

Built from a production conversational AI engagement framework ([LDTMP — Layered Depth Tension Modulation Protocol](https://github.com/amilovidov/meeting-debrief#research-foundations)) that monitors behavioral signals during voice conversations. The same signals that make an AI conversation feel human — response latency, vocabulary diversity, emotional keywords, pacing — turn out to be exactly the signals that reveal how a meeting actually went.

## Install

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) (for audio conversion)
- A [HuggingFace](https://huggingface.co/settings/tokens) account and token (free, for speaker diarization models)

### Accept model licenses (one-time, free)

Visit each link and click "Agree":

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### Install the package

**Apple Silicon (M1/M2/M3/M4) — recommended:**

Uses an MPS-optimized fork of pyannote for GPU-accelerated diarization.

```bash
pip install "meeting-debrief[mps] @ git+https://github.com/amilovidov/meeting-debrief.git"
```

**CPU / Linux / Windows:**

```bash
pip install "meeting-debrief[cpu] @ git+https://github.com/amilovidov/meeting-debrief.git"
```

## Usage

```bash
# Basic usage (auto-detects MPS/CUDA/CPU)
export HF_TOKEN="hf_your_token_here"
meeting-debrief recording.m4a

# Specify output directory and Whisper model
meeting-debrief recording.m4a -o ./output --whisper-model medium

# Known number of speakers (faster)
meeting-debrief recording.m4a --num-speakers 2

# Skip diarization (transcript analysis only)
meeting-debrief recording.m4a --skip-diarization

# Force CPU
meeting-debrief recording.m4a --device cpu
```

## Output

```
output/
├── recording_transcript.txt      # Timestamped transcript
├── recording_diarization.json    # Speaker segments with timestamps
├── recording_analysis.json       # Raw analysis data
└── recording_report.md           # Human-readable analysis report
```

## Performance

Tested on Apple M4 (iPad Pro / Mac):

| Audio Length | Diarization | Full Pipeline | Device |
|---|---|---|---|
| 5 min | 17s | ~1 min | M4 MPS |
| 70 min | 181s (~3 min) | ~10 min | M4 MPS |

Diarization runs at 0.04x realtime on Apple Silicon MPS. The bottleneck is Whisper transcription — use `--whisper-model base` for speed or `--whisper-model medium` for accuracy.

## How it works

### Architecture

```
Audio File (any format)
  ↓ ffmpeg
16kHz Mono WAV
  ↓ OpenAI Whisper (local)
Timestamped Transcript
  ↓ pyannote.audio (local, MPS/CUDA/CPU)
Speaker-Separated Segments
  ↓ librosa + Python analysis
Multi-Layer Analysis Report
```

### Analysis layers

The analysis framework is inspired by research into what makes conversations feel human vs mechanical. The same signals that an AI conversation engine monitors to adapt its behavior in real time are the signals that reveal conversation dynamics after the fact:

1. **Vocabulary diversity** drops 2-3 turns before someone disengages — it's an early warning system
2. **Response latency** (long pauses) often means processing, not disinterest — but unannounced pauses read as uncertainty
3. **Conviction/hedging ratio** reveals confidence independent of content — you can say the right thing with hedging words and still sound unsure
4. **Micro-pauses** within turns distinguish rehearsed material (fluent) from improvised answers (hesitant)
5. **Pitch expressiveness** separates animated engagement from monotone delivery

## Requirements

All dependencies are permissive open-source (MIT/BSD/ISC):

| Component | License | Purpose |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT | Local speech-to-text |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT | Speaker diarization |
| [librosa](https://github.com/librosa/librosa) | ISC | Audio feature extraction |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3 | Neural network runtime |
| [NumPy](https://github.com/numpy/numpy) | BSD-3 | Numerical computation |

The pyannote pretrained models are MIT licensed but gated on HuggingFace (free account required).

## Research Foundations

The analysis framework draws from established research in psycholinguistics, conversation analysis, and affective computing:

**Engagement signals:**
- Vocabulary diversity as a predictor of conversational disengagement — Gonzales, A. L., Hancock, J. T., & Pennebaker, J. W. (2010). *Language Style Matching as a Predictor of Social Dynamics in Small Groups.* Communication Research, 37(1), 3-19.
- Response latency as a cognitive load indicator — Brennan, S. E., & Williams, M. (1995). *The Feeling of Another's Knowing: Prosody and Filled Pauses as Cues to Listeners about the Metacognitive States of Speakers.* Journal of Memory and Language, 34(3), 383-398.
- Filler words ("um", "uh") as markers of speech planning complexity — Clark, H. H., & Fox Tree, J. E. (2002). *Using uh and um in Spontaneous Speaking.* Cognition, 84(1), 73-111.

**Emotional modeling:**
- Circumplex model of affect (valence + arousal dimensions) — Russell, J. A. (1980). *A Circumplex Model of Affect.* Journal of Personality and Social Psychology, 39(6), 1161-1178.

**Conversation dynamics:**
- Turn-taking and conversational structure — Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). *A Simplest Systematics for the Organization of Turn-Taking for Conversation.* Language, 50(4), 696-735.
- Narrative therapy and depth progression — White, M., & Epston, D. (1990). *Narrative Means to Therapeutic Ends.* W. W. Norton.
- Motivational interviewing techniques — Miller, W. R., & Rollnick, S. (2012). *Motivational Interviewing: Helping People Change* (3rd ed.). Guilford Press.

**Vocal analysis:**
- Pitch (F0) as an indicator of emotional arousal — Scherer, K. R. (2003). *Vocal Communication of Emotion: A Review of Research Paradigms.* Speech Communication, 40(1-2), 227-256.
- Micro-pauses and hesitation phenomena — Maclay, H., & Osgood, C. E. (1959). *Hesitation Phenomena in Spontaneous English Speech.* Word, 15(1), 19-44.

**LDTMP (Layered Depth Tension Modulation Protocol):**
The analysis framework is adapted from LDTMP, a conversational AI protocol developed by [Alexander Milovidov](https://linkedin.com/in/milovidov) for [Argo AI](https://getargoai.com), a voice-first storytelling platform. LDTMP monitors real-time engagement signals (response latency, vocabulary diversity, emotional valence, topic coherence) to adapt conversation depth through six progressive layers — from icebreaker to emotional saturation to reentry. The protocol uses deterministic code for phase transitions while the LLM handles natural language generation within each phase.

The key insight that motivated this tool: the same signals LDTMP uses to make AI conversations feel human are the signals that reveal how any conversation — meeting, interview, coaching session — actually went.

## License

MIT
