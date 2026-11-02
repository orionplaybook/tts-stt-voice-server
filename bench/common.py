"""Shared helpers for the benchmark scripts: memory probes and text normalization for WER."""

from __future__ import annotations

import ctypes
import os
import re
import unicodedata

_libc = ctypes.CDLL("libc.dylib", use_errno=True)


class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


def footprint(pid: int | None = None) -> tuple[float, float]:
    """(current, lifetime peak) physical footprint in GB: what Activity Monitor calls Memory,
    Metal buffers included (unlike RSS)."""
    info = _RUsageInfoV2()
    if _libc.proc_pid_rusage(pid or os.getpid(), 2, ctypes.byref(info)) != 0:
        return 0.0, 0.0
    # V2 has no lifetime max; V4 does but its layout is longer. Track the peak by sampling.
    cur = info.ri_phys_footprint / 1e9
    return cur, cur


class PeakSampler:
    """Samples this process's footprint every 20 ms in a thread; .peak is the max seen. Above
    limit_gb the process exits at once: a benchmark must never push the Mac into swap."""

    def __init__(self, limit_gb: float = 12.0):
        import threading
        self.limit_gb = limit_gb
        self.peak = footprint()[0]
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, footprint()[0])
            if self.peak > self.limit_gb:
                import sys
                print(f"memory guard: footprint {self.peak:.1f} GB > {self.limit_gb} GB, aborting", file=sys.stderr)
                os._exit(3)

    def reset(self):
        self.peak = footprint()[0]

    def stop(self):
        self._stop.set()
        self._t.join()
        return self.peak


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    # Keep letters (accents included), digits and apostrophes inside words; drop other punctuation.
    s = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) or ch == "'" else " " for ch in s)
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    r = [normalize_text(x) for x in refs]
    h = [normalize_text(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]
    return jiwer.wer([a for a, _ in pairs], [b if b else "<empty>" for _, b in pairs])


def hf_license(repo: str) -> str:
    """Licence tag of a Hugging Face repo (falls back to the base model's), '?' if unknown."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, expand=["cardData", "tags"])
        tags = [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")]
        card = info.card_data.to_dict() if info.card_data else {}
        lic = card.get("license_name") or card.get("license") or (tags[0] if tags else None)
        if lic in (None, "other") and card.get("base_model"):
            base = card["base_model"]
            base = base[0] if isinstance(base, list) else base
            if base != repo:
                return hf_license(base) + f" (from {base})"
        return lic or "?"
    except Exception:
        return "?"
