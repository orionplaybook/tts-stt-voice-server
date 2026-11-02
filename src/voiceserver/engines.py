"""Thin adapters over mlx-audio: one STT and one TTS entry point per model family.

mlx-audio does the inference. This module only smooths over the differences between model
families (how each takes a language, a voice, a speed) so the HTTP layer and the benchmarks
call every model the same way. Everything here is synchronous and must run on the single
inference thread (see worker.py): MLX inference is serialized per process.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STT_RATE = 16000

# ISO 639-1 -> language names, for model families that want names.
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ru": "Russian", "ar": "Arabic", "pl": "Polish", "sv": "Swedish", "tr": "Turkish",
}
# ISO 639-1 -> locale keys, for Nemotron's language prompt.
LANG_LOCALES = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "it": "it-IT",
                "pt": "pt-BR", "nl": "nl-NL", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN"}
# Kokoro's language codes.
KOKORO_LANGS = {"en": "a", "en-us": "a", "en-gb": "b", "fr": "f", "es": "e", "it": "i",
                "pt": "p", "ja": "j", "zh": "z", "hi": "h"}


def normalize_language(language: str | None) -> str | None:
    """'fr', 'FR', 'fr-CA', 'French' -> 'fr'; None or 'auto' -> None."""
    if not language:
        return None
    lang = language.strip().lower().replace("_", "-")
    if lang in ("", "auto"):
        return None
    for code, name in LANG_NAMES.items():
        if lang == name.lower():
            return code
    return lang.split("-")[0]


@dataclass
class Transcript:
    text: str
    language: str | None = None
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)


def stt_family(repo: str) -> str:
    r = repo.lower()
    for key in ("whisper", "parakeet", "qwen3-asr", "nemotron"):
        if key in r:
            return key
    return "generic"


def load_stt(repo: str, revision: str | None = None):
    from mlx_audio.stt.utils import load_model
    return load_model(repo, revision=revision) if revision else load_model(repo)


def split_at_pauses(audio: np.ndarray, max_s: float = 10.0, search_s: float = 4.0) -> list[tuple[int, int]]:
    """(start, end) sample ranges of at most max_s, each cut at the quietest 30 ms frame within the
    last search_s seconds of the window. Whisper turbo's long-form decoding skips or repeats whole
    sentences (measured on 60 s clips: 21-34% WER in English); independent ~10 s windows cut at
    pauses bring that down to about 9%. Only used for audio longer than 30 s."""
    n, win = audio.size, int(max_s * STT_RATE)
    if n <= win:
        return [(0, n)]
    frame = int(0.03 * STT_RATE)
    energy = np.sqrt(np.convolve(audio.astype(np.float64) ** 2, np.ones(frame) / frame, mode="same"))
    ranges, start = [], 0
    while n - start > win:
        lo, hi = start + win - int(search_s * STT_RATE), start + win
        cut = lo + int(np.argmin(energy[lo:hi]))
        ranges.append((start, cut))
        start = cut
    ranges.append((start, n))
    return ranges


def transcribe(model, repo: str, audio: np.ndarray, *, language: str | None = None,
               prompt: str | None = None, temperature: float | None = None,
               want_segments: bool = False) -> Transcript:
    """audio: mono float32 at 16 kHz."""
    family = stt_family(repo)
    if family == "whisper" and audio.size > 30 * STT_RATE:
        ranges = split_at_pauses(audio)
        parts = [_transcribe(model, family, audio[a:b], language, prompt, temperature, want_segments)
                 for a, b in ranges]
        segments = []
        for (a, _), p in zip(ranges, parts):
            segments += [{**s, "start": s["start"] + a / STT_RATE, "end": s["end"] + a / STT_RATE}
                         for s in p.segments]
        for i, s in enumerate(segments):
            s["id"] = i
        return Transcript(text=" ".join(p.text for p in parts if p.text), language=parts[0].language,
                          duration=audio.size / STT_RATE, segments=segments)
    return _transcribe(model, family, audio, language, prompt, temperature, want_segments)


def _transcribe(model, family: str, audio: np.ndarray, language: str | None, prompt: str | None,
                temperature: float | None, want_segments: bool) -> Transcript:
    import mlx.core as mx
    lang = normalize_language(language)
    x = mx.array(audio)
    kwargs: dict[str, Any] = {}
    if family == "whisper":
        kwargs.update(language=lang, initial_prompt=prompt or None, verbose=None,
                      word_timestamps=False, condition_on_previous_text=False)
        if temperature:  # 0 keeps whisper's own fallback ladder starting at 0
            kwargs["temperature"] = float(temperature)
    elif family == "qwen3-asr":
        kwargs.update(language=LANG_NAMES.get(lang, lang) if lang else None,
                      system_prompt=prompt or None)
    elif family == "nemotron":
        kwargs.update(language=LANG_LOCALES.get(lang, "auto") if lang else "auto")
    elif family == "parakeet":
        # Detects the language itself and takes no prompt. Unchunked, it drops whole sentences on
        # long audio (measured: 45% WER on 60 s clips); 30 s chunks fix that.
        kwargs.update(chunk_duration=30.0)
    result = model.generate(x, **kwargs)
    if hasattr(result, "__next__"):  # a generator: drain it
        result = list(result)[-1]
    text = (getattr(result, "text", None) or "").strip()
    detected = getattr(result, "language", None)
    if isinstance(detected, list):
        detected = detected[0] if detected else None
    segments = []
    if want_segments:
        for i, s in enumerate(_segments(result)):
            segments.append({"id": i, "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
                             "text": str(s.get("text", "")).strip()})
    return Transcript(text=text, language=normalize_language(detected) if detected else lang,
                      duration=audio.size / STT_RATE, segments=segments)


def _segments(result) -> list[dict[str, Any]]:
    segs = getattr(result, "segments", None)
    if segs:
        return [s if isinstance(s, dict) else {"start": getattr(s, "start", 0.0),
                                                "end": getattr(s, "end", 0.0),
                                                "text": getattr(s, "text", "")} for s in segs]
    sentences = getattr(result, "sentences", None)  # parakeet / nemotron AlignedResult
    if sentences:
        return [{"start": s.start, "end": s.end, "text": s.text} for s in sentences]
    return [{"start": 0.0, "end": 0.0, "text": getattr(result, "text", "")}]


# --- TTS ------------------------------------------------------------------------------------

@dataclass
class VoiceSpec:
    """How one named voice is produced by one model."""
    speaker: str | None = None          # native voice / speaker name
    ref_audio: str | None = None        # path to a reference clip (cloning models)
    ref_text: str | None = None
    instruct: str | None = None         # style instruction (models that take one)


def tts_family(repo: str) -> str:
    r = repo.lower()
    for key in ("kokoro", "chatterbox", "qwen3-tts", "voxcpm", "zonos"):
        if key in r:
            return key
    return "generic"


def load_tts(repo: str, revision: str | None = None):
    from mlx_audio.tts.utils import load_model
    return load_model(repo, revision=revision) if revision else load_model(repo)


def _tts_kwargs(repo: str, voice: VoiceSpec, language: str | None, speed: float,
                instructions: str | None) -> dict[str, Any]:
    family = tts_family(repo)
    lang = normalize_language(language)
    kw: dict[str, Any] = {"speed": speed, "verbose": False}
    if family == "kokoro":
        kw.update(voice=voice.speaker, lang_code=KOKORO_LANGS.get(lang or _kokoro_lang(voice.speaker), "a"))
    elif family == "chatterbox":
        kw.update(lang_code=lang or "en")
        if voice.ref_audio:
            kw["ref_audio"] = voice.ref_audio
    elif family == "qwen3-tts":
        kw.update(voice=voice.speaker, lang_code=LANG_NAMES.get(lang, "auto").lower() if lang else "auto")
        if instructions or voice.instruct:
            kw["instruct"] = instructions or voice.instruct
    elif family == "voxcpm":
        kw.pop("speed")
        if voice.ref_audio:
            kw.update(ref_audio=voice.ref_audio, ref_text=voice.ref_text)
        elif instructions or voice.instruct:
            kw["instruct"] = instructions or voice.instruct
    else:
        if voice.speaker:
            kw["voice"] = voice.speaker
        if voice.ref_audio:
            kw.update(ref_audio=voice.ref_audio, ref_text=voice.ref_text)
        if lang:
            kw["lang_code"] = lang
    return kw


def _kokoro_lang(speaker: str | None) -> str:
    # Kokoro voice names start with the language letter: af_heart (American), ff_siwis (French).
    first = (speaker or "a")[0]
    return {"a": "en", "b": "en-gb", "f": "fr", "e": "es", "i": "it", "p": "pt", "j": "ja", "z": "zh",
            "h": "hi"}.get(first, "en")


def native_voices(repo: str) -> list[str]:
    """Voice names the model itself ships (Kokoro: files under voices/ in its download)."""
    if tts_family(repo) != "kokoro":
        return []
    try:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo, local_files_only=True)) / "voices"
        return sorted({f.stem for f in path.iterdir() if f.suffix in (".safetensors", ".pt")})
    except Exception:
        return []


def supports_streaming(repo: str) -> bool:
    """Models that yield audio while a sentence is still being generated. (Multilingual
    Chatterbox accepts stream=True but ignores it; Chatterbox Turbo does stream.)"""
    r = repo.lower()
    return tts_family(repo) == "qwen3-tts" or "chatterbox-turbo" in r


def synthesize(model, repo: str, text: str, *, voice: VoiceSpec, language: str | None = None,
               speed: float = 1.0, instructions: str | None = None,
               stream: bool = False, streaming_interval: float = 0.6) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (float32 mono chunk, sample_rate). Without stream=True each chunk is a full segment
    (models split long text into sentences or lines themselves)."""
    kw = _tts_kwargs(repo, voice, language, speed, instructions)
    if stream and supports_streaming(repo):
        kw.update(stream=True, streaming_interval=streaming_interval)
    for result in model.generate(text=text, **kw):
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            yield audio, int(result.sample_rate)


def synthesize_text(model, repo: str, text: str, *, voice: VoiceSpec, language: str | None = None,
                    speed: float = 1.0, instructions: str | None = None,
                    stream: bool = True) -> Iterator[tuple[np.ndarray, int]]:
    """The service's speech path: sentence by sentence (and within a sentence, model-native
    streaming where the model has it), so the first audio comes after the first sentence."""
    from .text import split_sentences

    for sentence in split_sentences(text) if stream else [text]:
        yield from synthesize(model, repo, sentence, voice=voice, language=language, speed=speed,
                              instructions=instructions, stream=stream)


def model_sample_rate(model) -> int:
    return int(getattr(model, "sample_rate", 24000) or 24000)


def set_cache_limit(gb: float) -> None:
    """Cap MLX's buffer cache. Without a cap it keeps freed GPU buffers around and a 0.6B model
    can sit at 7-8 GB; with it, memory goes back to the system between requests."""
    import mlx.core as mx
    mx.set_cache_limit(int(gb * 2**30))


def clear_cache() -> None:
    import mlx.core as mx
    mx.clear_cache()


def release() -> None:
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()


def now() -> float:
    return time.perf_counter()
