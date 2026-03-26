"""Generate markdown analysis report from analysis results."""


def generate_report(results: dict, output_path: str) -> str:
    """Generate a markdown report from analysis results."""
    r = results
    spk = r["speakers"]
    lines = []

    lines.append("# Meeting Debrief Report")
    lines.append("")
    lines.append(f"**Primary speaker:** {spk['primary']} ({spk['primary_time']} min)")
    lines.append(f"**Secondary speaker:** {spk['secondary']} ({spk['secondary_time']} min)")
    lines.append("")

    # 1. Talk ratio
    lines.append("## Talk Ratio")
    lines.append("")
    lines.append(f"| Window | {spk['primary']} | {spk['secondary']} | Ratio |")
    lines.append("|---|---|---|---|")
    for w in r["talk_ratio"]:
        lines.append(f"| {w['window']} | {w['primary_seconds']}s | {w['secondary_seconds']}s | {w['ratio']}x |")
    lines.append("")

    # 2. Fillers
    lines.append("## Filler Words (cognitive load)")
    lines.append("")
    lines.append("| Window | Fillers | Words | Filler % | Top Fillers |")
    lines.append("|---|---|---|---|---|")
    for w in r["fillers"]:
        top = ", ".join(f"{k}({v})" for k, v in w["breakdown"].items())
        lines.append(f"| {w['window']} | {w['total_fillers']} | {w['total_words']} | {w['filler_pct']}% | {top} |")
    lines.append("")

    # 3. Vocabulary diversity
    lines.append("## Vocabulary Diversity (engagement signal)")
    lines.append("")
    lines.append("Higher = richer language, more engaged. Drop = repetitive or disengaged.")
    lines.append("")
    lines.append("| Window | Diversity | Unique/Total |")
    lines.append("|---|---|---|")
    for w in r["vocabulary_diversity"]:
        bar = "#" * int(w["diversity"] * 30)
        lines.append(f"| {w['window']} | {w['diversity']:.3f} | {w['unique']}/{w['total']} |")
    lines.append("")

    # 4. Conviction vs hedging
    lines.append("## Conviction vs Hedging")
    lines.append("")
    lines.append("C/H ratio >2x = assertive. <1x = uncertain.")
    lines.append("")
    lines.append("| Window | Emotional % | Hedging % | Conviction % | C/H Ratio | Signal |")
    lines.append("|---|---|---|---|---|---|")
    for w in r["conviction_hedging"]:
        signal = "confident" if w["ch_ratio"] >= 2 else ("balanced" if w["ch_ratio"] >= 1 else "uncertain")
        lines.append(f"| {w['window']} | {w['emotional_pct']}% | {w['hedging_pct']}% | {w['conviction_pct']}% | {w['ch_ratio']}x | {signal} |")
    lines.append("")

    # 5. Per-speaker pitch
    lines.append("## Per-Speaker Pitch (F0)")
    lines.append("")
    lines.append("Higher pitch = excitement/stress. Higher expressiveness = more animated.")
    lines.append("")
    lines.append(f"| Window | {spk['primary']} Hz | Std | Express% | {spk['secondary']} Hz | Std | Express% |")
    lines.append("|---|---|---|---|---|---|---|")
    for w in r["pitch"]:
        lines.append(
            f"| {w['window']} | {w['primary_avg_hz']} | {w['primary_std']} | {w['primary_expressiveness']}% "
            f"| {w['secondary_avg_hz']} | {w['secondary_std']} | {w['secondary_expressiveness']}% |"
        )
    lines.append("")

    # 6. Response latency
    rl = r["response_latency"]
    lines.append(f"## Response Latency ({spk['secondary']} finishes -> {spk['primary']} starts)")
    lines.append("")
    lines.append("| Window | Avg | Min | Max | Count | Signal |")
    lines.append("|---|---|---|---|---|---|")
    for w in rl["windows"]:
        signal = "quick" if w["avg"] < 0.8 else ("thinking" if w["avg"] < 1.5 else "slow/uncertain")
        lines.append(f"| {w['window']} | {w['avg']}s | {w['min']}s | {w['max']}s | {w['count']} | {signal} |")
    lines.append("")
    if rl["longest"]:
        lines.append("**Longest hesitations:**")
        for l in rl["longest"][:5]:
            ts = f"{int(l['timestamp']//60)}:{int(l['timestamp']%60):02d}"
            lines.append(f"- {l['latency']}s at {ts}")
        lines.append("")

    # 7. Turn duration
    lines.append("## Turn Duration Patterns")
    lines.append("")
    lines.append(f"| Window | {spk['primary']} Avg | Max | {spk['secondary']} Avg | Max |")
    lines.append("|---|---|---|---|---|")
    for w in r["turn_duration"]:
        lines.append(f"| {w['window']} | {w['primary_avg']}s | {w['primary_max']}s | {w['secondary_avg']}s | {w['secondary_max']}s |")
    lines.append("")

    # 8. Vocal energy
    lines.append("## Vocal Energy Per Speaker")
    lines.append("")
    lines.append(f"| Window | {spk['primary']} | {spk['secondary']} | Ratio |")
    lines.append("|---|---|---|---|")
    for w in r["vocal_energy"]:
        ratio = round(w["primary_energy"] / max(w["secondary_energy"], 0.0001), 1)
        lines.append(f"| {w['window']} | {w['primary_energy']} | {w['secondary_energy']} | {ratio}x |")
    lines.append("")

    # 9. Micro-pauses
    lines.append(f"## Micro-Pauses in {spk['primary']}'s Turns")
    lines.append("")
    lines.append("Internal pauses >0.3s within a speaking turn = mid-thought hesitation.")
    lines.append("")
    lines.append("| Window | Pauses | Turns | Rate/Min | Signal |")
    lines.append("|---|---|---|---|---|")
    for w in r["micro_pauses"]:
        signal = "fluent" if w["rate_per_min"] < 3 else ("some hesitation" if w["rate_per_min"] < 6 else "frequent hesitation")
        lines.append(f"| {w['window']} | {w['pauses']} | {w['turns']} | {w['rate_per_min']}/min | {signal} |")
    lines.append("")

    # 10. Engagement signals
    lines.append(f"## {spk['secondary']}'s Engagement Signals")
    lines.append("")
    lines.append("| Window | Short (<2s) | Medium (2-5s) | Long (>5s) | Signal |")
    lines.append("|---|---|---|---|---|")
    for w in r["engagement_signals"]:
        if w["long"] > w["short"]:
            signal = "deeply engaged"
        elif w["medium"] > w["short"]:
            signal = "engaged"
        else:
            signal = "listening"
        lines.append(f"| {w['window']} | {w['short']} | {w['medium']} | {w['long']} | {signal} |")
    lines.append("")

    # 11. Verbal tics
    if r["verbal_tics"]:
        lines.append("## Verbal Tics")
        lines.append("")
        for word, count in r["verbal_tics"].items():
            flag = " (noticeable)" if count > 15 else ""
            lines.append(f"- **\"{word}\"**: {count} times{flag}")
        lines.append("")

    # 12. Answer openers
    if r["answer_openers"]:
        lines.append("## Answer Opening Patterns")
        lines.append("")
        lines.append("| Pattern | Count |")
        lines.append("|---|---|")
        for pattern, count in r["answer_openers"].items():
            lines.append(f"| {pattern} | {count} |")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    return output_path
