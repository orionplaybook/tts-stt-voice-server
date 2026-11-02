"""Memory of the whole service, as deployed: starts the real server (default config) on a spare
port with a throwaway token, runs a workload through HTTP (voice notes, a 60 s clip, a paragraph
on each TTS tier), samples the server's physical footprint and the system's free memory, then
stops it. Adds LLM budgets given on the command line to show the headroom left.

  uv run python bench/memory_bench.py [--llm-gb 30 --llm-gb 17]    # -> bench/results/memory.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

from common import footprint  # noqa: E402

PORT = 18100


def system_memory() -> dict:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(vm.split("page size of ")[1].split()[0])
    pages = {line.split(":")[0].strip(): int(line.split(":")[1].strip().rstrip(".")) for line in vm.splitlines()[1:]
             if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()}
    avail = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0)
             + pages.get("Pages purgeable", 0)) * page
    return {"total_gb": round(total / 1e9, 1), "available_gb": round(avail / 1e9, 1),
            "wired_gb": round(pages.get("Pages wired down", 0) * page / 1e9, 1)}


def request(path: str, token: str, data: bytes | None = None, ctype: str | None = None) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Authorization": f"Bearer {token}", **({"Content-Type": ctype} if ctype else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def multipart(path: Path, fields: dict) -> tuple[bytes, str]:
    b = "----vsbench"
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-gb", type=float, action="append", default=[],
                    help="peak memory of an LLM that runs alongside (repeatable)")
    args = ap.parse_args()

    from voiceserver.tokens import TokenStore
    tmp = Path(tempfile.mkdtemp(prefix="vs-mem-"))
    base_cfg = (PROJECT / "config.toml").read_text()
    cfg_path = tmp / "config.toml"
    cfg_path.write_text(base_cfg.replace('port = 8100', f'port = {PORT}')
                        .replace('tokens_file = "~/.config/voice-server/tokens.json"', f'tokens_file = "{tmp}/tokens.json"')
                        .replace('log_dir = "~/Library/Logs/voice-server"', f'log_dir = "{tmp}/logs"'))
    for rel in ("voices",):  # voices referenced relative to the config
        if (PROJECT / rel).exists():
            os.symlink(PROJECT / rel, tmp / rel)
    token = TokenStore(tmp / "tokens.json").add("memory-bench")

    before = system_memory()
    proc = subprocess.Popen([sys.executable, "-m", "voiceserver", "--config", str(cfg_path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak, samples = 0.0, []
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.wait(0.05):
            cur = footprint(proc.pid)[0]
            peak = max(peak, cur)
            if cur > 16:  # never push the Mac into swap
                print(f"memory guard: server at {cur:.1f} GB, stopping it", flush=True)
                proc.kill()
                return
            samples.append(cur)
    threading.Thread(target=sample, daemon=True).start()
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None or time.time() - t0 > 600:
                    raise SystemExit("server did not start")
                time.sleep(1)
        startup_s = time.time() - t0
        idle_after_start = footprint(proc.pid)[0]
        clips = ROOT / "data" / "clips"
        passages = json.loads((ROOT / "passages.json").read_text())
        for _ in range(2):
            for name in ("en_voicenote.ogg", "fr_voicenote.ogg", "fr_60s.wav"):
                body, ctype = multipart(clips / name, {"model": "whisper-1"})
                request("/v1/audio/transcriptions", token, body, ctype)
            for model in ("tts-1", "tts-1-hd"):
                for lang in ("en", "fr"):
                    request("/v1/audio/speech", token, json.dumps(
                        {"model": model, "input": passages[lang]["paragraph"], "response_format": "mp3"}).encode(),
                        "application/json")
        time.sleep(3)
        steady = footprint(proc.pid)[0]
        during = system_memory()
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=30)

    llm = sum(args.llm_gb)
    result = {
        "date": time.strftime("%Y-%m-%d"), "startup_s": round(startup_s, 1),
        "server_footprint_gb": {"after_start": round(idle_after_start, 2), "peak_under_load": round(peak, 2),
                                "after_load": round(steady, 2)},
        "system_before": before, "system_during": during,
        "llm_budget_gb": args.llm_gb,
        "budget": None if not llm else {
            "llm_gb": llm, "voice_peak_gb": round(peak, 2), "total_gb": round(llm + peak, 1),
            "of_total_gb": before["total_gb"],
            "left_for_macos_and_apps_gb": round(before["total_gb"] - llm - peak, 1)},
    }
    out = ROOT / "results" / "memory.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
