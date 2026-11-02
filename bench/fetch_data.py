"""Fetch the public STT reference set used by the benchmarks.

Source: Google FLEURS (CC-BY-4.0), dev split, read speech with reference transcripts.
Only the first N utterances of each language archive are read; the download stops there.

Produces in bench/data/ (not in git, regenerate with this script):
  <lang>/utt/<id>.wav          16 kHz mono originals
  <lang>/refs.json             {id: transcript}
  clips/<lang>_10s.wav         one utterance close to 10 s
  clips/<lang>_60s.wav         consecutive utterances (several speakers) joined to ~60 s
  clips/<lang>_voicenote.ogg   the 10 s clip as Ogg/Opus 48 kHz mono, as Slack sends voice notes
  clips/clips.json             {clip: reference transcript}

Run:  uv run python bench/fetch_data.py
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LANGS = {"en": "en_us", "fr": "fr_fr", "de": "de_de"}
N_UTTS = 40
OUT = Path(__file__).resolve().parent / "data"


def fetch_refs(cfg: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{BASE}/{cfg}/dev.tsv", timeout=60) as r:
        text = r.read().decode("utf-8")
    refs: dict[str, str] = {}
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) >= 3:
            refs[row[1]] = row[2]  # file name -> raw transcription (cased, punctuated)
    return refs


def fetch_audio(cfg: str, wanted: set[str], dest: Path, limit: int) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    with urllib.request.urlopen(f"{BASE}/{cfg}/audio/dev.tar.gz", timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in wanted:
                    continue
                data = tar.extractfile(member).read()
                (dest / name).write_bytes(data)
                got.append(name)
                if len(got) >= limit:
                    break
    return got


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    clips_dir = OUT / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_refs: dict[str, str] = {}
    for lang, cfg in LANGS.items():
        print(f"{lang}: fetching references and {N_UTTS} utterances from FLEURS {cfg} dev")
        refs = fetch_refs(cfg)
        utt_dir = OUT / lang / "utt"
        names = fetch_audio(cfg, set(refs), utt_dir, N_UTTS)
        kept = {Path(n).stem: refs[n] for n in names}
        (OUT / lang / "refs.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1))

        durs = {n: duration(utt_dir / n) for n in names}
        # 10 s clip: the utterance closest to 10 s.
        ten = min(names, key=lambda n: abs(durs[n] - 10.0))
        ffmpeg("-i", str(utt_dir / ten), "-ac", "1", "-ar", "16000", str(clips_dir / f"{lang}_10s.wav"))
        clip_refs[f"{lang}_10s"] = refs[ten]
        # 60 s clip: consecutive utterances (different speakers) until >= 58 s, with 0.4 s gaps.
        chosen, total = [], 0.0
        for n in names:
            if n == ten:
                continue
            chosen.append(n)
            total += durs[n] + 0.4
            if total >= 58:
                break
        listing = clips_dir / f"{lang}_60s.txt"
        silence = clips_dir / "gap.wav"
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.4", str(silence))
        listing.write_text("".join(f"file '{utt_dir / n}'\nfile '{silence}'\n" for n in chosen))
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-ac", "1", "-ar", "16000",
               str(clips_dir / f"{lang}_60s.wav"))
        listing.unlink()
        silence.unlink()
        clip_refs[f"{lang}_60s"] = " ".join(refs[n] for n in chosen)
        # Voice note: Ogg/Opus 48 kHz mono, 32 kb/s, like Slack's.
        ffmpeg("-i", str(clips_dir / f"{lang}_10s.wav"), "-ac", "1", "-ar", "48000", "-c:a", "libopus",
               "-b:a", "32k", str(clips_dir / f"{lang}_voicenote.ogg"))
        clip_refs[f"{lang}_voicenote"] = refs[ten]
        print(f"  10 s clip {durs[ten]:.1f} s, 60 s clip {total:.1f} s from {len(chosen)} utterances")
    (clips_dir / "clips.json").write_text(json.dumps(clip_refs, ensure_ascii=False, indent=1))
    print(f"done: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
