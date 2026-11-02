"""STT benchmark: latency, real-time factor, word error rate and peak memory per model.

Each model runs in its own process (so its peak memory is its own), through the same
engine code the server uses (voiceserver.engines). Audio is decoded with ffmpeg exactly as
an upload would be, and the decode time is part of the measured latency.

  uv run python bench/stt_bench.py                     # every model in STT_MODELS
  uv run python bench/stt_bench.py --model <repo>      # one model (what the parent runs)

Needs bench/data from bench/fetch_data.py. Writes bench/results/stt.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from common import PeakSampler, footprint, wer  # noqa: E402

STT_MODELS = [
    # repo, licence
    ("mlx-community/whisper-large-v3-turbo-asr-fp16", "MIT"),
    ("mlx-community/parakeet-tdt-0.6b-v3", "CC-BY-4.0"),
    ("mlx-community/nemotron-3.5-asr-streaming-0.6b", "OpenMDW-1.1 (MLX copy: NVIDIA Open Model License)"),
    ("mlx-community/Qwen3-ASR-1.7B-8bit", "Apache-2.0"),
    ("mlx-community/Qwen3-ASR-0.6B-8bit", "Apache-2.0"),
]
LANGS = ["en", "fr", "de"]
CLIPS = ["10s", "60s", "voicenote"]
REPEATS = 3


def run_one(repo: str) -> dict:
    from voiceserver import audio, engines

    data = ROOT / "data"
    clip_refs = json.loads((data / "clips" / "clips.json").read_text())
    engines.set_cache_limit(1.0)  # as the service does
    sampler = PeakSampler()
    base = footprint()[0]
    t0 = time.perf_counter()
    from voiceserver.modellock import record_download
    record_download(repo)  # so `voicectl models clean` can remove it later
    model = engines.load_stt(repo)
    load_s = time.perf_counter() - t0
    # Warm-up (first call compiles kernels); not counted.
    engines.transcribe(model, repo, audio.decode((data / "clips" / "en_10s.wav").read_bytes()), language="en")

    out: dict = {"model": repo, "load_s": round(load_s, 2), "clips": {}, "wer": {}}
    for lang in LANGS:
        for clip in CLIPS:
            name = f"{lang}_{clip}"
            path = data / "clips" / (f"{name}.ogg" if clip == "voicenote" else f"{name}.wav")
            raw = path.read_bytes()
            times, text = [], ""
            for _ in range(REPEATS):
                t = time.perf_counter()
                pcm = audio.decode(raw)
                text = engines.transcribe(model, repo, pcm, language=lang).text
                times.append(time.perf_counter() - t)
            dur = pcm.size / engines.STT_RATE
            lat = statistics.median(times)
            out["clips"][name] = {"audio_s": round(dur, 1), "latency_s": round(lat, 3),
                                  "rtf": round(lat / dur, 4), "wer": round(wer([clip_refs[name]], [text]), 4),
                                  "text": text}
        # Reference set: every fetched utterance, language given.
        refs = json.loads((data / lang / "refs.json").read_text())
        hyps, refl = [], []
        for uid, ref in refs.items():
            pcm = audio.decode((data / lang / "utt" / f"{uid}.wav").read_bytes())
            hyps.append(engines.transcribe(model, repo, pcm, language=lang).text)
            refl.append(ref)
        out["wer"][lang] = {"utterances": len(refl), "wer": round(wer(refl, hyps), 4)}
        # Same set without a language hint (what happens when Hermes sends none).
        if lang != "de":
            hyps_auto = []
            for uid in refs:
                pcm = audio.decode((data / lang / "utt" / f"{uid}.wav").read_bytes())
                hyps_auto.append(engines.transcribe(model, repo, pcm, language=None).text)
            out["wer"][lang]["wer_auto_language"] = round(wer(refl, hyps_auto), 4)
    out["peak_footprint_gb"] = round(sampler.stop(), 2)
    out["base_footprint_gb"] = round(base, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=argparse.SUPPRESS)  # child process: one model
    ap.add_argument("--models", nargs="+", metavar="REPO", help="Hugging Face repos to benchmark "
                    "(default: the original candidate list)")
    args = ap.parse_args()
    if args.model:
        print(json.dumps(run_one(args.model), ensure_ascii=False))
        return
    from common import hf_license
    results_path = ROOT / "results" / "stt.json"
    results_path.parent.mkdir(exist_ok=True)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    todo = [(r, hf_license(r)) for r in args.models] if args.models else STT_MODELS
    for repo, lic in todo:
        print(f"== {repo}", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--model", repo], capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            results[repo] = {"model": repo, "license": lic, "error": proc.stderr.strip().splitlines()[-1:]}
        else:
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            r["license"] = lic
            results[repo] = r
            print(json.dumps({k: v for k, v in r.items() if k != "clips"}, ensure_ascii=False))
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"written {results_path}")


if __name__ == "__main__":
    main()
