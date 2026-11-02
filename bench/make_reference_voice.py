"""Create a synthetic reference voice for the cloning-only TTS models.

VoxCPM2 has no built-in speakers: without a reference clip it invents a new voice on every call,
which is useless for an assistant. So we design one voice ONCE from a text description (VoxCPM2's
voice design), keep that clip, and clone it on every call afterwards. No real person's voice.

  uv run python bench/make_reference_voice.py   -> bench/voices/reference.wav + reference.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

DESCRIPTION = ("A warm, clear adult female voice with a calm, friendly tone and a natural, "
               "unhurried conversational pace; studio quality, no background noise.")
TEXT = ("Hello, I'm your assistant. I can read your messages, keep track of your day, "
        "and answer questions whenever you need me.")


def main() -> None:
    from voiceserver import audio, engines

    out = ROOT / "voices"
    out.mkdir(exist_ok=True)
    model = engines.load_tts("mlx-community/VoxCPM2-bf16")
    chunks = [np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in
              model.generate(text=TEXT, instruct=DESCRIPTION, verbose=False)]
    sr = int(model.sample_rate)
    wav = np.concatenate(chunks)
    (out / "reference.wav").write_bytes(audio.encode(wav, sr, "wav"))
    (out / "reference.txt").write_text(TEXT + "\n")
    print(f"wrote {out / 'reference.wav'} ({wav.size / sr:.1f} s at {sr} Hz)")


if __name__ == "__main__":
    main()
