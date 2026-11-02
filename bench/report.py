"""Render the result tables used in BENCHMARKS.md from bench/results/*.json.

  uv run python bench/report.py      -> prints Markdown tables (STT, TTS, memory)
"""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def stt() -> str:
    r = json.loads((RES / "stt.json").read_text())
    rows = ["| Model | Licence | WER EN | WER FR | WER DE | WER EN/FR, no language hint | 10 s voice note (Ogg/Opus) EN / FR | 60 s clip EN / FR: latency, WER | Peak memory |",
            "|---|---|---|---|---|---|---|---|---|"]
    for repo, v in r.items():
        if "error" in v:
            rows.append(f"| {repo} | {v.get('license','')} | error: {v['error']} |||||||")
            continue
        w, c = v["wer"], v["clips"]
        rows.append(
            f"| `{repo.split('/')[-1]}` | {v['license']} | {pct(w['en']['wer'])} | {pct(w['fr']['wer'])} | "
            f"{pct(w['de']['wer'])} | {pct(w['en'].get('wer_auto_language'))} / {pct(w['fr'].get('wer_auto_language'))} | "
            f"{c['en_voicenote']['latency_s']:.2f} s / {c['fr_voicenote']['latency_s']:.2f} s | "
            f"{c['en_60s']['latency_s']:.2f} s, {pct(c['en_60s']['wer'])} / {c['fr_60s']['latency_s']:.2f} s, {pct(c['fr_60s']['wer'])} | "
            f"{v['peak_footprint_gb']:.1f} GB |")
    detail = ["", "Per clip (median of 3 runs; latency includes ffmpeg decoding; RTF = latency / audio length):", "",
              "| Model | Clip | Audio | Latency | RTF | WER |", "|---|---|---|---|---|---|"]
    for repo, v in r.items():
        for name, x in v.get("clips", {}).items():
            detail.append(f"| `{repo.split('/')[-1]}` | {name} | {x['audio_s']:.0f} s | {x['latency_s']:.2f} s | "
                          f"{x['rtf']:.3f} | {pct(x['wer'])} |")
    return "\n".join(rows + detail)


def tts() -> str:
    r = json.loads((RES / "tts.json").read_text())
    rows = ["| Sample | Model | Licence | Size | Languages | Native streaming | First audio, sentence EN / FR / DE | Sentence total EN / FR | Paragraph: first audio / total (EN) | Paragraph RTF EN / FR | Peak memory |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in sorted(r.items(), key=lambda kv: kv[1].get("label", "Z")):
        if "error" in v:
            rows.append(f"| | {key} | | | error: {v['error']} |||||||")
            continue
        runs = v["runs"]

        def g(k, f):
            return f"{runs[k][f]:.2f} s" if k in runs else "–"

        def rtf(k):
            return f"{runs[k]['rtf']:.2f}" if k in runs else "–"
        rows.append(
            f"| {v['label']} | `{v['repo'].split('/')[-1]}` | {v['license']} | {v.get('params','')} | {', '.join(v['languages'])} | "
            f"{'yes' if v['streaming'] else 'no (per sentence)'} | {g('en_sentence','ttfa_s')} / {g('fr_sentence','ttfa_s')} / {g('de_sentence','ttfa_s')} | "
            f"{g('en_sentence','total_s')} / {g('fr_sentence','total_s')} | {g('en_paragraph','ttfa_s')} / {g('en_paragraph','total_s')} | "
            f"{rtf('en_paragraph')} / {rtf('fr_paragraph')} | {v['peak_footprint_gb']:.1f} GB |")
    return "\n".join(rows)


def memory() -> str:
    p = RES / "memory.json"
    if not p.exists():
        return "(not measured yet)"
    m = json.loads(p.read_text())
    s = m["server_footprint_gb"]
    lines = [f"- Service footprint (all default models loaded and warm): {s['after_start']:.1f} GB after start, "
             f"{s['peak_under_load']:.1f} GB peak under load, {s['after_load']:.1f} GB after the load.",
             f"- Start-up to healthy: {m['startup_s']:.0f} s."]
    if m.get("budget"):
        b = m["budget"]
        lines.append(f"- With LLMs at {' + '.join(f'{x:g}' for x in m['llm_budget_gb'])} GB peak: "
                     f"{b['total_gb']:.1f} GB of {b['of_total_gb']:.0f} GB, leaving {b['left_for_macos_and_apps_gb']:.1f} GB "
                     f"for macOS, Docker and everything else.")
    return "\n".join(lines)


if __name__ == "__main__":
    print("## STT\n\n" + stt() + "\n\n## TTS\n\n" + tts() + "\n\n## Memory\n\n" + memory())
