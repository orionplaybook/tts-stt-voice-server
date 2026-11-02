"""Audio decoding and encoding through ffmpeg.

Every upload is decoded by ffmpeg (Ogg/Opus voice notes, m4a, mp3, wav, webm, flac...) to
mono float32 at the rate the model wants. Synthesized audio is encoded the same way. ffmpeg
runs as a subprocess with pipes, so no temp files are written for audio.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Iterator

import numpy as np

# OpenAI speech response formats -> (ffmpeg output args, content type)
SPEECH_FORMATS: dict[str, tuple[list[str], str]] = {
    "mp3": (["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"], "audio/ogg"),
    "aac": (["-f", "adts", "-c:a", "aac", "-b:a", "128k"], "audio/aac"),
    "flac": (["-f", "flac", "-c:a", "flac"], "audio/flac"),
    "wav": (["-f", "wav", "-c:a", "pcm_s16le"], "audio/wav"),
    "pcm": (["-f", "s16le", "-c:a", "pcm_s16le"], "audio/pcm"),
}
# OpenAI's own pcm format is 24 kHz; Hermes expects that unless told otherwise.
PCM_RATE = 24000
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


class AudioError(ValueError):
    """The upload could not be decoded as audio."""


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if shutil.which(p)), None)
    if not path:
        raise RuntimeError("ffmpeg not found on PATH (install it, e.g. `brew install ffmpeg`)")
    return path


def decode(data: bytes, sample_rate: int = 16000, timeout: float = 60.0,
           max_seconds: float | None = 30 * 60) -> np.ndarray:
    """Any container/codec ffmpeg understands -> mono float32 at sample_rate. Decoding stops at
    max_seconds (+1 s to detect overruns): a small, highly compressed upload can otherwise expand
    to hours of samples, gigabytes in memory, before any queue limit applies."""
    limit = ["-t", f"{max_seconds + 1:g}"] if max_seconds else []
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0",
         "-vn", "-ac", "1", "-ar", str(sample_rate), *limit, "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        reason = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no audio stream"]
        raise AudioError(f"could not decode audio: {reason[0]}")
    if max_seconds and len(proc.stdout) > max_seconds * sample_rate * 4:
        raise AudioError(f"audio longer than {max_seconds / 60:g} minutes")
    return np.frombuffer(proc.stdout, dtype=np.float32)  # read-only view, no second copy


def _out_rate(fmt: str, sample_rate: int) -> int:
    if fmt == "pcm":
        return PCM_RATE
    if fmt == "opus" and sample_rate not in OPUS_RATES:
        return 48000
    return sample_rate


def output_rate(fmt: str, sample_rate: int) -> int:
    """Sample rate of the encoded output for a model producing sample_rate."""
    return _out_rate(fmt, sample_rate)


def _encoder_args(fmt: str, sample_rate: int) -> list[str]:
    args, _ = SPEECH_FORMATS[fmt]
    return [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-ar", str(_out_rate(fmt, sample_rate)), "-ac", "1", *args, "pipe:1"]


def encode(samples: np.ndarray, sample_rate: int, fmt: str, timeout: float = 120.0) -> bytes:
    proc = subprocess.run(_encoder_args(fmt, sample_rate), input=_f32(samples).tobytes(),
                          capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg encode failed: " + proc.stderr.decode("utf-8", "replace")[-300:])
    return proc.stdout


def pcm16(samples: np.ndarray, sample_rate: int, target_rate: int = PCM_RATE) -> bytes:
    """Raw little-endian int16 mono, resampled in numpy (cheap enough for speech, no ffmpeg hop)."""
    x = _f32(samples)
    if sample_rate != target_rate and x.size:
        n = int(round(x.size * target_rate / sample_rate))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _f32(samples: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))


class StreamEncoder:
    """Feed float32 chunks in, read encoded bytes out while synthesis is still running."""

    def __init__(self, fmt: str, sample_rate: int):
        self._proc = subprocess.Popen(_encoder_args(fmt, sample_rate), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._out: list[bytes] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self._proc.stdout is not None
        while chunk := self._proc.stdout.read1(65536):
            with self._lock:
                self._out.append(chunk)

    def feed(self, samples: np.ndarray) -> bytes:
        assert self._proc.stdin is not None
        self._proc.stdin.write(_f32(samples).tobytes())
        self._proc.stdin.flush()
        return self.take()

    def take(self) -> bytes:
        with self._lock:
            out, self._out = b"".join(self._out), []
        return out

    def close(self) -> bytes:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        self._reader.join(timeout=30)
        self._proc.wait(timeout=30)
        return self.take()

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait(timeout=5)


def iter_chunks(samples: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for i in range(0, samples.size, size):
        yield samples[i:i + size]
