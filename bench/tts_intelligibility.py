"""Intelligibility check for the TTS samples: transcribe every samples/*.mp3 with Whisper and
compare with the text it was given. Catches dropped sentences, garbled words and wrong-language
pronunciation; it says nothing about how natural a voice sounds (that is the listening test).

  uv run python bench/tts_intelligibility.py   -> bench/results/tts_intelligibility.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from common import wer  # noqa: E402

REPO = "mlx-community/whisper-large-v3-turbo-asr-fp16"


def main() -> None:
    from voiceserver import audio, engines

    passages = json.loads((ROOT / "passages.json").read_text())
    engines.set_cache_limit(1.0)
    model = engines.load_stt(REPO)
    out: dict[str, dict] = {}
    for f in sorted((ROOT.parent / "samples").glob("*_*_*.mp3")):
        label, lang, kind = f.stem.split("_")
        text = engines.transcribe(model, REPO, audio.decode(f.read_bytes()), language=lang).text
        out.setdefault(label, {})[f"{lang}_{kind}"] = {"wer": round(wer([passages[lang][kind]], [text]), 4),
                                                       "heard": text}
    (ROOT / "results" / "tts_intelligibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for label, runs in out.items():
        print(label, {k: v["wer"] for k, v in runs.items()})


if __name__ == "__main__":
    main()
