"""TTS benchmark and blind listening samples.

For every candidate: time to first audio and total time for one sentence and one paragraph per
language, through the service's own speech path (sentence splitting plus model-native streaming
where the model has it), and peak memory. Each model runs in its own process.

The rendered passages are saved to samples/ under a letter (A, B, C...) with no model name;
samples/KEY.md maps letters to models. The letter order is shuffled once and kept in
bench/results/tts_labels.json so re-runs keep the same letters.

  uv run python bench/tts_bench.py                 # every candidate
  uv run python bench/tts_bench.py --only kokoro   # one candidate (key below)

Needs bench/voices/ references for the cloning models (bench/make_reference_voice.py).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

from common import PeakSampler, footprint  # noqa: E402

VOICES_DIR = ROOT / "voices"
REF = {"ref_audio": str(VOICES_DIR / "reference.wav"), "ref_text_file": str(VOICES_DIR / "reference.txt")}

CANDIDATES = {
    "kokoro": {
        "repo": "mlx-community/Kokoro-82M-bf16", "license": "Apache-2.0", "params": "82M",
        "voices": {"en": {"speaker": "af_heart"}, "fr": {"speaker": "ff_siwis"}},
    },
    "chatterbox": {
        "repo": "mlx-community/chatterbox-fp16", "license": "MIT", "params": "0.5B",
        "voices": {"en": {}, "fr": {}, "de": {}},
    },
    "chatterbox-8bit": {
        "repo": "mlx-community/chatterbox-8bit", "license": "MIT", "params": "0.5B (8-bit)",
        "voices": {"en": {}, "fr": {}, "de": {}},
    },
    "chatterbox-turbo": {
        "repo": "mlx-community/chatterbox-turbo-fp16", "license": "MIT", "params": "350M",
        "voices": {"en": {}},
    },
    "qwen3-tts": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", "license": "Apache-2.0",
        "params": "1.7B", "voices": {"en": {"speaker": "aiden"}, "fr": {"speaker": "aiden"},
                                      "de": {"speaker": "aiden"}},
    },
    # VoxCPM2 was measured once (2026-11-02) and dropped: slower than real time (RTF 1.1-1.3) and
    # memory grew past 50 GB on a 64 GB Mac. Its samples (letter A) stay for the listening test.
    "voxcpm2": {"dropped": True,
        "repo": "mlx-community/VoxCPM2-bf16", "license": "Apache-2.0", "params": "2B",
        "voices": {"en": REF, "fr": REF, "de": REF},
    },
}


EXTRA = ROOT / "results" / "tts_candidates.json"  # candidates added with --repo


def all_candidates() -> dict:
    extra = json.loads(EXTRA.read_text()) if EXTRA.exists() else {}
    return {**CANDIDATES, **extra}


def add_candidate(repo: str, voice: str | None, languages: list[str]) -> str:
    from common import hf_license
    key = repo.split("/")[-1].lower()
    extra = json.loads(EXTRA.read_text()) if EXTRA.exists() else {}
    spec = {"speaker": voice} if voice else {}
    extra[key] = {"repo": repo, "license": hf_license(repo), "params": "",
                  "voices": {lang: spec for lang in languages}}
    EXTRA.parent.mkdir(exist_ok=True)
    EXTRA.write_text(json.dumps(extra, indent=1) + "\n")
    return key


def voice_spec(d: dict):
    from voiceserver.engines import VoiceSpec
    ref_text = Path(d["ref_text_file"]).read_text().strip() if d.get("ref_text_file") else None
    return VoiceSpec(speaker=d.get("speaker"), ref_audio=d.get("ref_audio"), ref_text=ref_text)


def run_one(key: str, label: str) -> dict:
    from voiceserver import audio, engines

    cand = all_candidates()[key]
    repo = cand["repo"]
    passages = json.loads((ROOT / "passages.json").read_text())
    engines.set_cache_limit(1.0)  # as the service does
    sampler = PeakSampler()
    t0 = time.perf_counter()
    from voiceserver.modellock import record_download
    record_download(repo)  # so `voicectl models clean` can remove it later
    model = engines.load_tts(repo)
    load_s = time.perf_counter() - t0
    out = {"key": key, "repo": repo, "label": label, "load_s": round(load_s, 1),
           "streaming": engines.supports_streaming(repo), "runs": {}}
    if hasattr(model, "get_supported_speakers"):
        out["speakers"] = model.get_supported_speakers()
    first_lang = next(iter(cand["voices"]))
    # Warm-up: compile kernels, load language pipelines.
    for lang, v in cand["voices"].items():
        list(engines.synthesize_text(model, repo, passages[lang]["sentence"][:40], voice=voice_spec(v),
                                     language=lang))
    samples_dir = PROJECT / "samples"
    samples_dir.mkdir(exist_ok=True)
    for lang, v in cand["voices"].items():
        for kind in ("sentence", "paragraph"):
            text = passages[lang][kind]
            chunks, sr, ttfa = [], 24000, None
            t = time.perf_counter()
            for chunk, sr in engines.synthesize_text(model, repo, text, voice=voice_spec(v), language=lang):
                if ttfa is None:
                    ttfa = time.perf_counter() - t
                chunks.append(chunk)
            total = time.perf_counter() - t
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            dur = wav.size / sr
            (samples_dir / f"{label}_{lang}_{kind}.mp3").write_bytes(audio.encode(wav, sr, "mp3"))
            out["runs"][f"{lang}_{kind}"] = {"ttfa_s": round(ttfa or total, 2), "total_s": round(total, 2),
                                             "audio_s": round(dur, 1), "rtf": round(total / max(dur, 0.01), 3),
                                             "sample_rate": sr}
    out["peak_footprint_gb"] = round(sampler.stop(), 2)
    out["languages"] = list(cand["voices"])
    del first_lang
    return out


def labels() -> dict[str, str]:
    path = ROOT / "results" / "tts_labels.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    keys = [k for k in all_candidates() if k not in existing]
    free = [chr(ord("A") + i) for i in range(26) if chr(ord("A") + i) not in existing.values()]
    random.Random(len(existing) + 7).shuffle(keys)
    for k, letter in zip(keys, free):
        existing[k] = letter
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(existing, indent=1))
    return existing


def write_key(lab: dict[str, str], results: dict) -> None:
    lines = ["# Listening test key", "",
             "Listen to the samples first, by letter, and pick your favourites before opening this.", "",
             "| Letter | Model | Licence | Languages |", "|---|---|---|---|"]
    for key, letter in sorted(lab.items(), key=lambda kv: kv[1]):
        c = all_candidates().get(key, {})
        langs = ", ".join(results.get(key, {}).get("languages", c.get("voices", {}).keys()))
        note = " (dropped: slower than real time, too much memory)" if c.get("dropped") else ""
        lines.append(f"| {letter} | {c.get('repo', key)}{note} | {c.get('license', '?')} | {langs} |")
    (PROJECT / "samples" / "KEY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--label")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repo", help="benchmark this Hugging Face repo as a new candidate")
    ap.add_argument("--voice", help="with --repo: the model's speaker/voice name, if it has named voices")
    ap.add_argument("--languages", default="en,fr", help="with --repo: comma-separated (default en,fr)")
    args = ap.parse_args()
    if args.one:
        print(json.dumps(run_one(args.one, args.label)))
        return
    if args.repo:
        args.only = [add_candidate(args.repo, args.voice, [x.strip() for x in args.languages.split(",") if x.strip()])]
    lab = labels()
    path = ROOT / "results" / "tts.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in args.only or [k for k, c in all_candidates().items() if not c.get("dropped")]:
        print(f"== {key} ({lab[key]})", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--one", key, "--label", lab[key]],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
            results[key] = {"key": key, "error": proc.stderr.strip().splitlines()[-1:]}
        else:
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            r["license"] = all_candidates()[key]["license"]
            results[key] = r
            print(json.dumps(r))
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    write_key(lab, results)
    print(f"written {path} and samples/")


if __name__ == "__main__":
    main()
