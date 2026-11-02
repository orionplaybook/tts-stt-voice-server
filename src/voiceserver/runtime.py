"""Model lifecycle and the inference queue.

MLX inference is serialized per process, so every model call runs on ONE worker thread. Each
model has its own admission limit (max_queue: requests waiting or running for it); beyond that
a request gets 429 at once rather than piling up. A request that waited longer than
queue_timeout_s for the thread is dropped before it runs. Streamed speech is generated one chunk
per job, so a transcription can slip in between two chunks of a long reply.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from . import engines, modellock
from .config import Config, ModelConfig

log = logging.getLogger("voiceserver")
T = TypeVar("T")


class Busy(Exception):
    """Too many requests queued for this model (HTTP 429)."""


class QueueTimeout(Exception):
    """Waited too long for the inference thread (HTTP 503)."""


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._models: dict[str, Any] = {}          # name -> loaded model (touched on the mlx thread only)
        self._inflight: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self._reaper: asyncio.Task | None = None

    # --- lifecycle ----------------------------------------------------------------------

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(
            self._executor, engines.set_cache_limit, self.cfg.server.mlx_cache_limit_gb)
        for m in self.cfg.models.values():
            if m.preload:
                t = time.perf_counter()
                await self._submit(m, lambda _model: None, admit=False)
                log.info("loaded %s (%s) in %.1f s", m.name, m.repo, time.perf_counter() - t)
        self._reaper = asyncio.create_task(self._reap_idle())

    async def stop(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def loaded(self) -> list[str]:
        return sorted(self._models)

    def _ensure(self, m: ModelConfig):  # mlx thread
        model = self._models.get(m.name)
        if model is None:
            rev = modellock.revision(m.repo)  # the pinned commit (models.lock.json), if any
            model = engines.load_stt(m.repo, rev) if m.kind == "stt" else engines.load_tts(m.repo, rev)
            self._models[m.name] = model
            self._warm(m, model)
        return model

    def _warm(self, m: ModelConfig, model) -> None:  # mlx thread: compile kernels before real traffic
        import numpy as np
        try:
            if m.kind == "stt":
                engines.transcribe(model, m.repo, np.zeros(engines.STT_RATE, dtype=np.float32), language="en")
            else:
                for lang in (m.languages or ["en"])[:3]:
                    spec = self.voice_for(m, m.default_voice, lang)
                    for _ in engines.synthesize(model, m.repo, "Ready.", voice=spec, language=lang):
                        pass
        except Exception:  # warm-up is best effort
            log.warning("warm-up of %s failed", m.name, exc_info=True)

    def _unload(self, name: str) -> None:  # mlx thread
        if self._models.pop(name, None) is not None:
            engines.release()
            log.info("unloaded %s after idle timeout", name)

    async def _reap_idle(self) -> None:
        while True:
            await asyncio.sleep(15)
            now = time.monotonic()
            for m in self.cfg.models.values():
                if (m.idle_unload_s > 0 and m.name in self._models and not self._inflight.get(m.name)
                        and now - self._last_used.get(m.name, now) > m.idle_unload_s):
                    await asyncio.get_running_loop().run_in_executor(self._executor, self._unload, m.name)

    # --- running jobs -------------------------------------------------------------------

    def voice_for(self, m: ModelConfig, voice: str | None, lang: str | None) -> engines.VoiceSpec:
        """Configured voice (or OpenAI alias) -> spec for this language; unknown names are passed
        through as the model's native speaker name."""
        name = voice or m.default_voice
        if name not in m.voices:
            name = self.cfg.voice_aliases.get(name, name)
        if name not in m.voices:
            return engines.VoiceSpec(speaker=name or None)
        table = m.voices[name]
        return table.get(lang or "") or table.get("*") or next(iter(table.values()))

    def _admit(self, m: ModelConfig) -> None:
        n = self._inflight.get(m.name, 0)
        if n >= m.max_queue:
            raise Busy(f"model {m.name!r} has {n} requests queued; try again shortly")
        self._inflight[m.name] = n + 1

    def _release(self, m: ModelConfig) -> None:
        self._inflight[m.name] = max(0, self._inflight.get(m.name, 1) - 1)
        self._last_used[m.name] = time.monotonic()

    async def _submit(self, m: ModelConfig, fn: Callable[[Any], T], admit: bool = True,
                      enqueued: float | None = None) -> T:
        enqueued = enqueued or time.monotonic()
        queue_timeout = self.cfg.server.queue_timeout_s

        def job():
            if admit and time.monotonic() - enqueued > queue_timeout:
                raise QueueTimeout(f"waited more than {queue_timeout:.0f} s for the model")
            return fn(self._ensure(m))

        def job_and_trim():
            try:
                return job()
            finally:
                if self.cfg.server.release_after_request:
                    engines.clear_cache()

        fut = asyncio.get_running_loop().run_in_executor(self._executor, job_and_trim)
        return await asyncio.wait_for(fut, timeout=self.cfg.server.request_timeout_s + queue_timeout)

    async def run(self, m: ModelConfig, fn: Callable[[Any], T]) -> T:
        self._admit(m)
        try:
            return await self._submit(m, fn)
        finally:
            self._release(m)

    async def stream(self, m: ModelConfig, make: Callable[[Any], Iterator[T]]) -> AsyncIterator[T]:
        """Run a generator on the mlx thread one item per job. Admission is held until the
        generator is exhausted or the consumer stops (client disconnect)."""
        self._admit(m)
        gen_box: list[Iterator[T]] = []
        done = object()

        def first(model):
            gen_box.append(make(model))
            return next(gen_box[0], done)

        try:
            item = await self._submit(m, first)
            while item is not done:
                yield item
                item = await self._submit(m, lambda _model: next(gen_box[0], done), enqueued=time.monotonic())
        finally:
            if gen_box:
                close = getattr(gen_box[0], "close", None)
                if close:
                    self._executor.submit(close)
            self._release(m)
