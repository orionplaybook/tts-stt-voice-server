"""Resident memory per model (loaded and warm, cache released) and the accuracy cost of
quantized variants. Each variant runs in its own process with the 12 GB guard.

  uv run python bench/resident.py              -> bench/results/resident.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

VARIANTS = [
    ("stt", "mlx-community/whisper-large-v3-turbo-asr-fp16"),
    ("stt", "mlx-community/whisper-large-v3-turbo-asr-6bit"),
    ("stt", "mlx-community/whisper-large-v3-turbo-asr-4bit"),
    ("stt", "mlx-community/Qwen3-ASR-1.7B-8bit"),
    ("tts", "mlx-community/Kokoro-82M-bf16"),
    ("tts", "mlx-community/chatterbox-fp16"),
    ("tts", "mlx-community/chatterbox-8bit"),
    ("tts", "mlx-community/chatterbox-4bit"),
    ("tts", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"),
]


def one(kind: str, repo: str) -> dict:
    import time

    from common import PeakSampler, footprint, wer
    from voiceserver import audio, engines

    from voiceserver.modellock import record_download
    record_download(repo)
    engines.set_cache_limit(0.25)
    sampler = PeakSampler()
    base = footprint()[0]
    out: dict = {"kind": kind, "repo": repo}
    if kind == "stt":
        model = engines.load_stt(repo)
        engines.transcribe(model, repo, audio.decode((ROOT / "data/clips/en_10s.wav").read_bytes()), language="en")
        engines.clear_cache()
        out["resident_gb"] = round(footprint()[0] - base, 2)
        for lang in ("en", "fr"):
            refs = json.loads((ROOT / "data" / lang / "refs.json").read_text())
            hyps = [engines.transcribe(model, repo, audio.decode((ROOT / "data" / lang / "utt" / f"{u}.wav").read_bytes()),
                                       language=lang).text for u in refs]
            out[f"wer_{lang}"] = round(wer(list(refs.values()), hyps), 4)
        t = time.perf_counter()
        engines.transcribe(model, repo, audio.decode((ROOT / "data/clips/fr_voicenote.ogg").read_bytes()), language="fr")
        out["voicenote_s"] = round(time.perf_counter() - t, 2)
    else:
        model = engines.load_tts(repo)
        passages = json.loads((ROOT / "passages.json").read_text())
        voice = engines.VoiceSpec(speaker={"kokoro": "af_heart", "qwen3-tts": "aiden"}.get(engines.tts_family(repo)))
        list(engines.synthesize_text(model, repo, "Ready to go.", voice=voice, language="en"))
        engines.clear_cache()
        out["resident_gb"] = round(footprint()[0] - base, 2)
        stt_repo = "mlx-community/whisper-large-v3-turbo-asr-fp16"
        heard = []
        for lang in ("en", "fr"):
            v = voice if engines.tts_family(repo) != "kokoro" else engines.VoiceSpec(speaker={"en": "af_heart", "fr": "ff_siwis"}[lang])
            t = time.perf_counter()
            chunks = [c for c, _ in engines.synthesize_text(model, repo, passages[lang]["paragraph"], voice=v, language=lang)]
            out[f"paragraph_{lang}_s"] = round(time.perf_counter() - t, 2)
            heard.append((lang, chunks))
        del model
        engines.release()
        stt = engines.load_stt(stt_repo)
        import numpy as np
        for lang, chunks in heard:
            wav = np.concatenate(chunks)
            sr = 24000
            text = engines.transcribe(stt, stt_repo, audio.decode(audio.encode(wav, sr, "wav")), language=lang).text
            out[f"intelligibility_wer_{lang}"] = round(wer([passages[lang]["paragraph"]], [text]), 4)
    out["peak_gb"] = round(sampler.stop() - base, 2)
    return out


def main() -> None:
    if len(sys.argv) == 3:
        print(json.dumps(one(sys.argv[1], sys.argv[2])))
        return
    path = ROOT / "results" / "resident.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for kind, repo in VARIANTS:
        print(f"== {repo}", flush=True)
        p = subprocess.run([sys.executable, __file__, kind, repo], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-1500:])
            results[repo] = {"kind": kind, "repo": repo, "error": p.stderr.strip().splitlines()[-1:]}
        else:
            results[repo] = json.loads(p.stdout.strip().splitlines()[-1])
            print(results[repo], flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
