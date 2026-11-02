"""voicectl: one command for everything you can do with voice-server.

  voicectl status                      is it running, which models are loaded
  voicectl say "Bonjour !"             speak text (plays it; -o file.mp3 saves it)
  voicectl transcribe note.m4a         print the transcript of an audio file
  voicectl tokens add|list|revoke      manage consumer tokens
  voicectl voices | config            what is configured
  voicectl models [discover|add|remove|delete|clean]   list, find, add and remove models
  voicectl bench stt|tts|memory|report benchmark models
  voicectl selftest                    live round trip in every language
  voicectl update [--check]            update packages and models (tested, rolled back on failure)
  voicectl autoupdate enable|disable|status   weekly automatic update
  voicectl start|stop|restart|logs|check|install|upgrade|uninstall

Run `voicectl <command> --help` for the options of each command.

Talking to the service needs a token. voicectl uses, in order: --token, $VOICE_SERVER_TOKEN, or
its own token, which it creates on first use (consumer "voicectl") and keeps next to the token
store, readable by you only. The server address comes from the config, or --url /
$VOICE_SERVER_URL (for example http://other-mac:8100).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .config import PROJECT_DIR, load_config
from .tokens import TokenStore

CLI_CONSUMER = "voicectl"
SCRIPTS = PROJECT_DIR / "scripts"


class CliError(Exception):
    pass


# --- HTTP ---------------------------------------------------------------------------------

class Client:
    """Minimal HTTP client on the standard library (the service environment has no extras)."""

    def __init__(self, base_url: str, token: str | None, timeout: float = 300, cafile: str | None = None):
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout
        self.tls = None
        if self.base_url.startswith("https://") and cafile:
            import ssl
            # Trust the server's own (possibly self-signed) certificate; on loopback its name
            # will not match 127.0.0.1, which is fine for a connection that never leaves the Mac.
            self.tls = ssl.create_default_context(cafile=cafile)
            if urllib.parse.urlsplit(self.base_url).hostname in ("127.0.0.1", "::1", "localhost"):
                self.tls.check_hostname = False

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        h = dict(headers or {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base_url + path, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.tls) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise CliError(f"cannot reach voice-server at {self.base_url}: {getattr(e, 'reason', e)}\n"
                           f"  Is it running?  voicectl status") from None

    def json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        status, _, data = self.request(method, path, body, {"Content-Type": "application/json"} if body else None)
        check(status, data)
        return json.loads(data)


def check(status: int, data: bytes) -> None:
    if status < 400:
        return
    try:
        message = json.loads(data)["error"]["message"]
    except Exception:
        message = data.decode("utf-8", "replace")[:300]
    hint = {401: "\n  The token was refused. Check --token / $VOICE_SERVER_TOKEN, or remove the saved "
                 "voicectl token so a new one is created."}.get(status, "")
    raise CliError(f"HTTP {status}: {message}{hint}")


def multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = "----voicectl" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --- context ------------------------------------------------------------------------------

def base_url(args) -> str:
    if args.url or os.environ.get("VOICE_SERVER_URL"):
        url = (args.url or os.environ["VOICE_SERVER_URL"]).rstrip("/")
        return url[:-3] if url.endswith("/v1") else url
    from .config import base_url as config_base_url
    return config_base_url(load_config(args.config).server)


def saved_token_path(args) -> Path:
    return load_config(args.config).server.tokens_file.with_name("voicectl.token")


def token_for(args) -> str:
    if args.token:
        return args.token
    if os.environ.get("VOICE_SERVER_TOKEN"):
        return os.environ["VOICE_SERVER_TOKEN"]
    path = saved_token_path(args)
    if path.exists():
        return path.read_text().strip()
    # First use on this Mac: create the CLI's own consumer token and keep it, private.
    store = TokenStore(load_config(args.config).server.tokens_file)
    if any(e["name"] == CLI_CONSUMER for e in store.names()):
        store.revoke(CLI_CONSUMER)  # its saved copy is gone: replace it
    token = store.add(CLI_CONSUMER)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    print(f"(created voicectl's own token, consumer {CLI_CONSUMER!r}, saved in {path})", file=sys.stderr)
    return token


def client(args, auth: bool = True) -> Client:
    return Client(base_url(args), token_for(args) if auth else None, timeout=args.timeout, cafile=_cafile(args))


def _cafile(args) -> str | None:
    if args.url or os.environ.get("VOICE_SERVER_URL"):
        return None  # another server: the system's trust store
    return load_config(args.config).server.tls_cert or None


def run_script(name: str, *extra: str) -> int:
    return subprocess.call(["/bin/bash", str(SCRIPTS / name), *extra])


# --- commands -----------------------------------------------------------------------------

def cmd_status(args) -> int:
    url = base_url(args)
    label = os.environ.get("VOICE_SERVER_LABEL", "local.voice-server")
    out = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{label}"], capture_output=True, text=True)
    if out.returncode == 0:
        info = {k.strip(): v.strip() for line in out.stdout.splitlines()
                if line.startswith("\t") and not line.startswith("\t\t") and " = " in line
                for k, v in [line.split(" = ", 1)]}
        print(f"launchd agent {label}: {info.get('state', '?')}"
              + (f", pid {info['pid']}" if "pid" in info else "")
              + (f", started {info.get('runs', '?')} time(s)" if "runs" in info else ""))
    else:
        print(f"launchd agent {label}: not loaded (voicectl install, or voicectl start)")
    try:
        status, _, data = Client(url, None, timeout=5, cafile=_cafile(args)).request("GET", "/health")
    except CliError:
        print(f"service: not answering on {url}")
        return 1
    health = json.loads(data) if status == 200 else {}
    print(f"service: {url}  {health.get('status', f'HTTP {status}')}, loaded: "
          f"{', '.join(health.get('models', [])) or 'none'}")
    return 0 if status == 200 else 1


def cmd_say(args) -> int:
    text = args.text
    if args.file:
        text = Path(args.file).read_text()
    elif text in (None, "-"):
        text = sys.stdin.read()
    text = (text or "").strip()
    if not text:
        raise CliError("nothing to say: give TEXT, --file, or pipe text in")
    fmt = args.format or (Path(args.output).suffix.lstrip(".").replace("ogg", "opus") if args.output else "mp3")
    payload = {"input": text, "response_format": fmt, "speed": args.speed}
    for key in ("model", "voice", "language", "instructions"):
        if getattr(args, key):
            payload[key] = getattr(args, key)
    c = client(args)
    t = time.perf_counter()
    status, headers, data = c.request("POST", "/v1/audio/speech", json.dumps(payload).encode(),
                                      {"Content-Type": "application/json"})
    check(status, data)
    took = time.perf_counter() - t
    if args.output:
        Path(args.output).write_bytes(data)
        print(f"saved {args.output} ({len(data) / 1024:.0f} KB, {fmt}, {took:.1f} s)", file=sys.stderr)
    if not args.output or args.play:
        if fmt == "pcm":
            raise CliError("pcm cannot be played directly; use -o file.pcm or another --format")
        player = shutil.which("afplay")
        if not player:
            raise CliError("no audio player (afplay) found; use -o FILE to save the audio")
        with tempfile.NamedTemporaryFile(suffix="." + ("ogg" if fmt == "opus" else fmt)) as f:
            f.write(data)
            f.flush()
            subprocess.run([player, f.name], check=False)
    if args.verbose:
        print(f"{len(text)} characters, {took:.2f} s, {headers.get('x-audio-sample-rate', '?')} Hz", file=sys.stderr)
    return 0


def cmd_transcribe(args) -> int:
    c = client(args)
    rc = 0
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            print(f"error: {name}: no such file", file=sys.stderr)
            rc = 1
            continue
        fields = {"response_format": args.format}
        for key in ("model", "language", "prompt"):
            if getattr(args, key):
                fields[key] = getattr(args, key)
        body, ctype = multipart(fields, "file", path)
        t = time.perf_counter()
        status, _, data = c.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": ctype})
        check(status, data)
        took = time.perf_counter() - t
        out = data.decode("utf-8")
        if args.format == "json":
            out = json.loads(out)["text"] + "\n"
        elif args.format == "verbose_json":
            out = json.dumps(json.loads(out), ensure_ascii=False, indent=1) + "\n"
        if args.output:
            target = Path(args.output) if len(args.files) == 1 else Path(args.output) / (path.stem + _ext(args.format))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(out)
            print(f"{name} -> {target} ({took:.1f} s)", file=sys.stderr)
        else:
            if len(args.files) > 1:
                print(f"== {name}")
            sys.stdout.write(out if out.endswith("\n") else out + "\n")
        if args.verbose:
            print(f"({took:.2f} s)", file=sys.stderr)
    return rc


def _ext(fmt: str) -> str:
    return {"srt": ".srt", "vtt": ".vtt", "verbose_json": ".json"}.get(fmt, ".txt")


def cmd_voices(args) -> int:
    data = client(args).json("GET", "/v1/audio/voices" + (f"?model={args.model}" if args.model else ""))["data"]
    for m in data:
        print(f"{m['model']}  (requested as: {', '.join([m['model']] + m['aliases'])})")
        print(f"  languages: {', '.join(m['languages']) or 'any'}"
              + (f"; other languages go to {m['fallback']}" if m.get("fallback") else ""))
        for v, per_lang in m["voices"].items():
            mark = " (default)" if v == m["default_voice"] else ""
            print(f"  voice {v}{mark}: " + ", ".join(f"{lang}={who}" for lang, who in per_lang.items()))
        if m["openai_voice_names"]:
            print(f"  OpenAI names that map to these: {', '.join(m['openai_voice_names'])}")
        if m["native_voices"] and (args.all or len(m["native_voices"]) <= 12):
            print(f"  its own voices ({len(m['native_voices'])}): {', '.join(m['native_voices'])}")
        elif m["native_voices"]:
            print(f"  its own voices: {len(m['native_voices'])} (voicectl voices --all lists them)")
    return 0


def cmd_models(args) -> int:
    return {"list": models_list, "discover": models_discover, "add": models_add, "remove": models_remove,
            "delete": models_delete, "clean": models_clean}[args.models_cmd or "list"](args)


def models_list(args) -> int:
    from . import modellock
    cfg = load_config(args.config)
    try:
        loaded = set(json.loads(Client(base_url(args), None, timeout=5, cafile=_cafile(args)).request("GET", "/health")[2])["models"])
        state_of = lambda n: "loaded" if n in loaded else "not loaded"  # noqa: E731
    except (CliError, ValueError, KeyError):
        state_of = lambda n: "service not running"  # noqa: E731
    sizes = modellock.cached_repos()
    pins = modellock.load()
    for kind in ("stt", "tts"):
        default = cfg.stt_default if kind == "stt" else cfg.tts_default
        for m in (m for m in cfg.models.values() if m.kind == kind):
            names = ", ".join([a for a, t in cfg.aliases.items() if t == m.name]) or "-"
            pin = pins.get(m.repo, {})
            size = sum(sizes.get(r, 0) for r in [m.repo, *m.extra_repos]) / 1e9
            when = "preloaded" if m.preload else (f"on demand, unloads after {m.idle_unload_s:g} s" if m.idle_unload_s else "on demand")
            print(f"{kind.upper()} {m.name}{' (default)' if m.name == default else ''}  [{state_of(m.name)}]")
            print(f"    {m.repo} @ {pin.get('revision', 'not pinned')[:10]} ({pin.get('date', '?')}), {size:.1f} GB, {when}")
            print(f"    also answers to: {names}" + (f"; speaks {', '.join(m.languages)}" if m.languages else "")
                  + (f"; other languages -> {m.fallback}" if m.fallback else ""))
    extra = sorted(r for r in modellock.downloaded() & set(sizes) if r not in modellock.required_repos(cfg))
    if extra:
        total = sum(sizes[r] for r in extra) / 1e9
        print(f"\nDownloaded but not used ({total:.1f} GB): {', '.join(extra)}\n    remove with: voicectl models clean")
    return 0


SPEECH_TASKS = {"automatic-speech-recognition": "stt", "text-to-speech": "tts", "text-to-audio": "tts"}


def models_discover(args) -> int:
    """Recent MLX speech models on Hugging Face, newest first."""
    from huggingface_hub import HfApi

    from . import modellock
    cfg = load_config(args.config)
    configured = {m.repo for m in cfg.models.values()}
    cached = modellock.cached_repos()
    cutoff = time.time() - args.days * 86400
    seen = 0
    print(f"MLX speech models from {args.author} updated in the last {args.days:g} days"
          + (f" matching {args.search!r}" if args.search else "") + ":")
    for m in HfApi().list_models(author=args.author, search=args.search, sort="last_modified", limit=args.scan,
                                 expand=["lastModified", "downloads", "pipeline_tag", "tags"]):
        kind = SPEECH_TASKS.get(m.pipeline_tag or "")
        if not kind or (args.kind and kind != args.kind) or m.last_modified.timestamp() < cutoff:
            continue
        if m.pipeline_tag == "text-to-audio" and not args.include_audio:
            continue  # mostly music generation and codecs, not speech
        lic = next((t.split(":", 1)[1] for t in (m.tags or []) if t.startswith("license:")), "?")
        flag = " CHECK LICENCE" if lic in ("?", "other") or "nc" in lic.lower() else ""
        mark = " [configured]" if m.id in configured else (" [downloaded]" if m.id in cached else "")
        music = " (text-to-audio: often music, not speech)" if m.pipeline_tag == "text-to-audio" else ""
        print(f"  {m.last_modified:%Y-%m-%d}  {kind.upper()}  {m.id}{mark}  {m.downloads or 0} downloads, "
              f"licence {lic}{flag}{music}")
        seen += 1
    if not seen:
        print("  none")
    print("\nTry one:  voicectl bench stt REPO   or   voicectl bench tts REPO [--voice NAME]")
    print("Use one:  voicectl models add NAME REPO   (then voicectl restart)")
    return 0


MARK_START, MARK_END = "# >>> voicectl models add {name}", "# <<< voicectl models add {name}"


def _local_config(args) -> Path:
    cfg = load_config(args.config)
    return cfg.source[0].with_name("config.local.toml")


def models_add(args) -> int:
    from . import modellock
    name, repo = args.name, args.repo
    cfg = load_config(args.config)
    if name in cfg.models or name in cfg.aliases:
        raise CliError(f"{name!r} is already a model or alias name (voicectl models)")
    kind = args.kind
    lic = "?"
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, expand=["pipeline_tag", "tags"])
        kind = kind or SPEECH_TASKS.get(info.pipeline_tag or "")
        lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), "?")
    except Exception as e:  # noqa: BLE001
        raise CliError(f"cannot find {repo} on Hugging Face: {e}") from None
    if kind not in ("stt", "tts"):
        raise CliError("cannot tell whether this is speech-to-text or text-to-speech; add --kind stt|tts")
    langs = [x.strip() for x in (args.languages or "").split(",") if x.strip()]
    lines = [MARK_START.format(name=name) + f" ({time.strftime('%Y-%m-%d')}, licence: {lic})",
             f"[{kind}.models.{name}]", f'repo = "{repo}"',
             f"preload = {'true' if args.preload else 'false'}",
             f"idle_unload_s = {0 if args.preload else 600}", "max_queue = 8"]
    if args.extra_repo:
        lines.append("extra_repos = [" + ", ".join(f'"{r}"' for r in args.extra_repo) + "]")
    if kind == "tts":
        lines += [f"languages = [{', '.join(repr(x).replace(chr(39), chr(34)) for x in langs)}]", 'default_voice = "default"',
                  f"[tts.models.{name}.voices.default]",
                  '"*" = { speaker = "%s" }' % args.voice if args.voice else '"*" = {}']
    lines.append(MARK_END.format(name=name))
    path = _local_config(args)
    before = path.read_text() if path.exists() else ""
    path.write_text(before.rstrip("\n") + ("\n\n" if before else "") + "\n".join(lines) + "\n")
    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        path.write_text(before)
        raise CliError(f"the new config would be invalid ({e}); nothing changed") from None
    if lic in ("?", "other") or "nc" in lic.lower():
        print(f"warning: licence of {repo} is {lic!r}: check it allows your use before relying on it", file=sys.stderr)
    print(f"added {kind.upper()} model {name!r} ({repo}) to {path.name}; downloading and pinning:")
    modellock.sync(cfg)
    print(f"\nNext: voicectl restart, then e.g. voicectl "
          + (f'transcribe FILE --model {name}' if kind == "stt" else f'say "Hello" --model {name}')
          + f"\nRemove it again with: voicectl models remove {name}")
    return 0


def _confirm(args, question: str) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        raise CliError("needs confirmation; add --yes")
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _unpin_unused(cfg) -> list[str]:
    from . import modellock
    pins = modellock.load()
    gone = [r for r in pins if r not in modellock.required_repos(cfg)]
    for r in gone:
        pins.pop(r)
    modellock.save(pins)
    return gone


def models_remove(args) -> int:
    from . import modellock
    name = args.name
    cfg = load_config(args.config)
    if name not in cfg.models:
        raise CliError(f"no model named {name!r} (voicectl models)")
    users = [a for a, t in cfg.aliases.items() if t == name]
    users += [f"default {k}" for k, d in (("stt", cfg.stt_default), ("tts", cfg.tts_default)) if d == name]
    users += [f"fallback of {m.name}" for m in cfg.models.values() if m.fallback == name]
    if users:
        raise CliError(f"{name!r} is still used ({', '.join(users)}); point those elsewhere first")
    path = _local_config(args)
    text = path.read_text() if path.exists() else ""
    start = text.find(MARK_START.format(name=name))
    end = text.find(MARK_END.format(name=name))
    if start < 0 or end < 0:
        raise CliError(f"{name!r} is defined in config.toml, not added with voicectl; remove it there by hand, "
                       "then run: voicectl models clean")
    repo = cfg.models[name].repo
    if not _confirm(args, f"Remove model {name!r} ({repo}) from the config"
                    + ("" if args.keep_files else " and delete its downloaded files") + "?"):
        return 1
    end = text.index("\n", end) + 1 if "\n" in text[end:] else len(text)
    path.write_text((text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")).strip("\n") + "\n")
    cfg = load_config(args.config)
    _unpin_unused(cfg)
    print(f"removed {name!r} from {path.name}")
    if not args.keep_files:
        if repo not in modellock.required_repos(cfg):  # another model may use the same repo
            print(f"deleted {repo} ({modellock.delete_cached(repo) / 1e9:.1f} GB)")
        # Helper repos (extra_repos) may be shared; `voicectl models clean` removes them once unused.
    print("restart to apply: voicectl restart")
    return 0


def models_delete(args) -> int:
    from . import modellock
    cfg = load_config(args.config)
    repo = args.repo
    if repo in modellock.required_repos(cfg):
        raise CliError(f"{repo} is used by the configuration; remove the model first (voicectl models remove NAME)")
    size = modellock.cached_repos().get(repo)
    if size is None:
        raise CliError(f"{repo} is not in the Hugging Face cache")
    if not _confirm(args, f"Delete {repo} ({size / 1e9:.1f} GB) from the Hugging Face cache?"):
        return 1
    print(f"deleted {repo} ({modellock.delete_cached(repo) / 1e9:.1f} GB freed)")
    return 0


def models_clean(args) -> int:
    from . import modellock
    cfg = load_config(args.config)
    sizes = modellock.cached_repos()
    unused = sorted(r for r in modellock.downloaded() & set(sizes) if r not in modellock.required_repos(cfg))
    if not unused:
        print("nothing to clean: every model downloaded for voice-server is in use")
        return 0
    for r in unused:
        print(f"  {sizes[r] / 1e9:5.2f} GB  {r}")
    total = sum(sizes[r] for r in unused) / 1e9
    if not _confirm(args, f"Delete these {len(unused)} unused models ({total:.1f} GB)?"):
        return 1
    for r in unused:
        modellock.delete_cached(r)
    print(f"freed {total:.1f} GB (only models downloaded by voice-server or its benchmarks are ever cleaned)")
    return 0


def cmd_bench(args) -> int:
    """Run the benchmark scripts (bench/), guarded against memory pressure."""
    bench = PROJECT_DIR / "bench"
    py = sys.executable
    if args.bench_cmd == "report":
        return subprocess.call([py, str(bench / "report.py")])
    avail = _available_memory_gb()
    if avail < args.min_free_gb and not args.force:
        raise CliError(f"only {avail:.0f} GB of memory available (need {args.min_free_gb:g} GB so a benchmark cannot "
                       "push the Mac into swap). Close something, or --force.")
    if args.bench_cmd in ("stt", "tts") and not (bench / "data" / "clips" / "clips.json").exists():
        print("fetching the public test audio first (FLEURS, ~40 MB)")
        if subprocess.call([py, str(bench / "fetch_data.py")]):
            return 1
    if args.bench_cmd == "stt":
        cfg = load_config(args.config)
        repos = args.repos or [m.repo for m in cfg.models.values() if m.kind == "stt"]
        rc = subprocess.call([py, str(bench / "stt_bench.py"), "--models", *repos])
    elif args.bench_cmd == "tts":
        rc = subprocess.call([py, str(bench / "tts_bench.py"), "--repo", args.repo, "--languages", args.languages,
                              *(["--voice", args.voice] if args.voice else [])])
        if rc == 0:
            rc = subprocess.call([py, str(bench / "tts_intelligibility.py")])
            print(f"\nListening samples: {PROJECT_DIR / 'samples'} (letters; key in samples/KEY.md)")
    else:  # memory
        rc = subprocess.call([py, str(bench / "memory_bench.py"), *[a for g in args.llm_gb for a in ("--llm-gb", str(g))]])
    if rc == 0 and args.bench_cmd != "memory":
        print("\nSummary tables: voicectl bench report   (BENCHMARKS.md is written by hand from them)")
    return rc


def _available_memory_gb() -> float:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(vm.split("page size of ")[1].split()[0])
    pages = {ln.split(":")[0].strip(): int(ln.split(":")[1].strip().rstrip(".")) for ln in vm.splitlines()[1:]
             if ":" in ln and ln.split(":")[1].strip().rstrip(".").isdigit()}
    return sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")) * page / 1e9


def cmd_tokens(args) -> int:
    store = TokenStore(load_config(args.config).server.tokens_file)
    if args.tokens_cmd == "add":
        try:
            token = store.add(args.name)
        except ValueError as e:
            raise CliError(str(e)) from None
        print(f"Token for {args.name!r} (shown once; store it with that program):", file=sys.stderr)
        print(token)
    elif args.tokens_cmd == "list":
        names = store.names()
        for e in names:
            print(f"{e['name']:30} created {e['created']}")
        if not names:
            print(f"no tokens yet ({store.path})")
    elif args.tokens_cmd == "revoke":
        if not store.revoke(args.name):
            raise CliError(f"no token named {args.name!r} (voicectl tokens list)")
        if args.name == CLI_CONSUMER:
            saved_token_path(args).unlink(missing_ok=True)
        print(f"revoked {args.name!r}; it stops working immediately")
    if warning := store.check_permissions():
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config)
    print("config files: " + ", ".join(str(p) for p in cfg.source))
    s = cfg.server
    print(f"listens on:   {s.host}:{s.port}")
    print(f"tokens:       {s.tokens_file}")
    print(f"logs:         {s.log_dir}{'  (content logging ON)' if s.log_content else ''}")
    print(f"limits:       upload {s.max_upload_mb:g} MB, speech input {s.max_input_chars} chars, "
          f"queue wait {s.queue_timeout_s:g} s")
    for kind in ("stt", "tts"):
        default = cfg.stt_default if kind == "stt" else cfg.tts_default
        for m in (m for m in cfg.models.values() if m.kind == kind):
            names = [a for a, t in cfg.aliases.items() if t == m.name]
            extra = []
            if m.preload:
                extra.append("preloaded")
            if m.idle_unload_s:
                extra.append(f"unloads after {m.idle_unload_s:g} s idle")
            if m.languages:
                extra.append("speaks " + "/".join(m.languages))
            if m.fallback:
                extra.append(f"fallback {m.fallback}")
            print(f"{kind.upper()} {m.name}{' (default)' if m.name == default else ''}: {m.repo}"
                  f"\n      names: {', '.join([m.name] + names)}; {', '.join(extra) or 'loads on demand'}")
    print("edit config.toml (or config.local.toml), then: voicectl restart")
    return 0


def cmd_script(args) -> int:
    mapping = {"start": ("voice-server.sh", ["start"]), "stop": ("voice-server.sh", ["stop"]),
               "restart": ("voice-server.sh", ["restart"]), "check": ("health-check.sh", []),
               "install": ("install.sh", []), "uninstall": ("uninstall.sh", []), "upgrade": ("upgrade.sh", []),
               "logs": ("voice-server.sh", ["logs", str(getattr(args, "lines", 50))])}
    script, extra = mapping[args.cmd]
    if args.cmd == "install" and args.no_start:
        extra = ["--no-start"]
    if args.cmd == "uninstall" and args.purge:
        extra = ["--purge"]
    if args.cmd == "upgrade" and args.check:
        extra = ["--check"]
    return run_script(script, *extra)


SELFTEST_TEXT = {"en": "The weather is lovely today, let us go for a walk in the park.",
                 "fr": "Il fait très beau aujourd'hui, allons nous promener dans le parc.",
                 "de": "Das Wetter ist heute schön, lass uns im Park spazieren gehen."}


def _words(s: str) -> set[str]:
    return {w.strip(".,!?;:'\"«»").lower() for w in s.split()} - {""}


def cmd_selftest(args) -> int:
    """Live check of the running service: health, auth, and for every language the default
    voice speaks (fallback included) a round trip text -> speech -> text."""
    url = base_url(args)
    ok = True

    def report(good: bool, name: str, detail: str) -> None:
        nonlocal ok
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {detail}")

    status, _, data = Client(url, None, timeout=10, cafile=_cafile(args)).request("GET", "/health")
    report(status == 200, "health", data.decode()[:120] if status == 200 else f"HTTP {status}")
    status, _, _ = Client(url, None, timeout=10, cafile=_cafile(args)).request("GET", "/v1/models")
    report(status == 401, "token required", f"HTTP {status} without a token")
    c = client(args)
    voices = c.json("GET", "/v1/audio/voices")["data"]
    default = next((m for m in voices if "tts-1-hd" in m["aliases"]), voices[0])
    langs = list(default["languages"])
    fb = next((m for m in voices if m["model"] == default.get("fallback")), None)
    langs += [x for x in (fb["languages"] if fb else []) if x not in langs and x in SELFTEST_TEXT]
    for lang in [x for x in langs if x in SELFTEST_TEXT]:
        text = SELFTEST_TEXT[lang]
        t = time.perf_counter()
        status, _, audio = c.request("POST", "/v1/audio/speech", json.dumps(
            {"model": "tts-1-hd", "voice": "alloy", "input": text, "response_format": "opus", "language": lang}).encode(),
            {"Content-Type": "application/json"})
        if status != 200:
            report(False, f"speech {lang}", f"HTTP {status}: {audio[:150]!r}")
            continue
        tts_s = time.perf_counter() - t
        with tempfile.NamedTemporaryFile(suffix=".ogg") as f:
            f.write(audio)
            f.flush()
            body, ctype = multipart({"model": "whisper-1", "language": lang, "response_format": "text"}, "file", Path(f.name))
        t = time.perf_counter()
        status, _, heard = c.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": ctype})
        stt_s = time.perf_counter() - t
        heard_text = heard.decode("utf-8", "replace").strip()
        overlap = len(_words(heard_text) & _words(text)) / len(_words(text))
        report(status == 200 and overlap >= 0.6, f"round trip {lang}",
               f"speech {tts_s:.2f} s, transcription {stt_s:.2f} s, {overlap:.0%} of words back")
    print("self-test passed" if ok else "self-test FAILED")
    return 0 if ok else 1


def cmd_update(args) -> int:
    from .updater import Updater
    u = Updater(args.config, cooldown_days=args.cooldown_days,
                log=lambda m: print(m, flush=True))
    if args.check:
        changes = u.check()
        for p in changes["packages"]:
            print(f"package {p['name']}: {p['old'] or '-'} -> {p['new'] or 'removed'}"
                  + (f" (released {p['released']})" if p.get("released") else ""))
        for m in changes["models"]:
            print(f"model {m['repo']}: {(m['old'] or '-')[:10]} -> {m['new'][:10]} ({m['date']})")
        for w in changes["waiting"]:
            print(f"waiting (released {w['released']}; younger than {args.cooldown_days:g} days, or date unknown): "
                  f"{w['name']} {w['old']} -> {w['new']}")
        if not changes["packages"] and not changes["models"]:
            print(f"up to date (counting releases at least {args.cooldown_days:g} days old)")
        return 0
    status = u.apply(restart=not args.no_restart)
    if status["git_uncommitted"]:
        print(f"note: {', '.join(status['git_uncommitted'])} changed; commit them to keep the new versions in git")
    return 0 if status["result"] in ("updated", "up-to-date") else 1


def cmd_autoupdate(args) -> int:
    from . import updater
    if args.action == "enable":
        when = updater.enable_schedule(args.weekday, args.time, args.cooldown_days, load_config(args.config).server.log_dir)
        print(f"automatic updates on: {when}, releases at least {args.cooldown_days:g} days old.")
        print("results: voicectl autoupdate status")
    elif args.action == "disable":
        print("automatic updates off" if updater.disable_schedule() else "automatic updates were not on")
    else:
        desc = updater.schedule_description()
        print(f"automatic updates: {desc or 'off (voicectl autoupdate enable)'}")
        path = load_config(args.config).server.log_dir / "update-status.json"
        if path.exists():
            st = json.loads(path.read_text())
            print(f"last run: {st.get('finished', '?')}: {st.get('result')}"
                  + (f", {len(st['packages'])} package(s), {len(st['models'])} model(s) changed"
                     if st.get("result") in ("updated", "rolled-back") else ""))
            if st.get("error"):
                print("  error: " + st["error"].splitlines()[0][:200])
            if st.get("git_uncommitted"):
                print(f"  not yet committed: {', '.join(st['git_uncommitted'])}")
        else:
            print("last run: never")
    return 0


# --- argument parsing ---------------------------------------------------------------------

def _common(top: bool) -> argparse.ArgumentParser:
    """Global options, accepted before or after the command. On subcommands they default to
    SUPPRESS so they do not overwrite a value given before the command."""
    p = argparse.ArgumentParser(add_help=False)
    d = (lambda v: v) if top else (lambda v: argparse.SUPPRESS)
    p.add_argument("--url", default=d(None), help="server address, e.g. http://127.0.0.1:8100 (default: from the config)")
    p.add_argument("--token", default=d(None), help="API token (default: $VOICE_SERVER_TOKEN, else voicectl's own)")
    p.add_argument("--config", default=d(None), help="config file (default: config.toml in the project)")
    p.add_argument("--timeout", type=float, default=d(300.0), help="seconds to wait for the server (default 300)")
    p.add_argument("-v", "--verbose", action="store_true", default=d(False), help="also print timings")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common(top=False)
    ap = argparse.ArgumentParser(
        prog="voicectl", parents=[_common(top=True)], formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Control and use the local voice-server (speech to text and text to speech).",
        epilog="""examples:
  voicectl status
  voicectl say "Hello, this is a test."
  voicectl say "Bonjour ! Ça marche." --voice ff_siwis -o bonjour.mp3
  echo "Long text..." | voicectl say - --model chatterbox
  voicectl transcribe voice-note.ogg
  voicectl transcribe meeting.m4a --language fr --format srt -o meeting.srt
  voicectl tokens add n8n
  voicectl voices
  voicectl selftest
  voicectl models discover --kind tts --days 30
  voicectl bench tts mlx-community/SOME-NEW-TTS --languages en,fr
  voicectl models add parakeet mlx-community/parakeet-tdt-0.6b-v3
  voicectl models clean
  voicectl update --check
  voicectl autoupdate enable --weekday sun --time 02:30
  voicectl logs""")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    p = sub.add_parser("status", parents=[common], help="is the service running, which models are loaded")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("say", parents=[common], help="speak text: play it, or save it with -o",
                       description="Turn text into speech. Plays it through the speakers unless -o is given.")
    p.add_argument("text", nargs="?", help='text to speak; "-" or nothing reads it from stdin')
    p.add_argument("-f", "--file", help="read the text from this file")
    p.add_argument("-o", "--output", help="save the audio here (format from the extension: .mp3 .ogg .wav .flac .aac)")
    p.add_argument("--play", action="store_true", help="play it even when saving with -o")
    p.add_argument("-m", "--model", help="tts-1, tts-1-hd, kokoro, chatterbox... (default: the configured default)")
    p.add_argument("--voice", help="alloy, nova... or a model's own voice such as af_bella (see: voicectl voices)")
    p.add_argument("-l", "--language", help="en, fr, de... (default: detected from the text)")
    p.add_argument("--speed", type=float, default=1.0, help="0.25 to 4 (default 1)")
    p.add_argument("--format", choices=["mp3", "opus", "aac", "flac", "wav", "pcm"], help="audio format (default mp3)")
    p.add_argument("--instructions", help="style instruction, for models that take one")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("transcribe", parents=[common], help="audio file(s) to text",
                       description="Transcribe audio files (ogg/opus voice notes, m4a, mp3, wav, webm, flac...).")
    p.add_argument("files", nargs="+", help="audio files")
    p.add_argument("-l", "--language", help="en, fr, de... (default: detected per file)")
    p.add_argument("-m", "--model", help="whisper-1 or a configured STT model (default: the configured default)")
    p.add_argument("--prompt", help="words to expect (names, jargon): improves spelling")
    p.add_argument("--format", default="text", choices=["text", "json", "verbose_json", "srt", "vtt"],
                   help="output: text (default), json, verbose_json (with segments), srt or vtt subtitles")
    p.add_argument("-o", "--output", help="write here instead of printing (a folder when several files)")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("voices", parents=[common], help="voices per TTS model and how to ask for them")
    p.add_argument("-m", "--model", help="only this model")
    p.add_argument("--all", action="store_true", help="list every native voice name")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("models", parents=[common], help="list, discover, add, remove or clean up models",
                       description="Without an action: list the configured models.")
    p.set_defaults(func=cmd_models, models_cmd=None)
    msub = p.add_subparsers(dest="models_cmd", metavar="action")
    msub.add_parser("list", parents=[common], help="configured models: names, pinned version, size, loaded or not")
    q = msub.add_parser("discover", parents=[common], help="recent MLX speech models on Hugging Face")
    q.add_argument("--kind", choices=["stt", "tts"])
    q.add_argument("--days", type=float, default=90, help="updated within this many days (default 90)")
    q.add_argument("--search", help="text in the repo name, e.g. whisper, kokoro")
    q.add_argument("--author", default="mlx-community", help="Hugging Face account (default mlx-community)")
    q.add_argument("--scan", type=int, default=500, help="how many recent repos to look through (default 500)")
    q.add_argument("--include-audio", action="store_true", help="also list text-to-audio repos (mostly music)")
    q = msub.add_parser("add", parents=[common], help="add a model to config.local.toml, download and pin it",
                        description="Adds an on-demand model (loads on first use, unloads after 10 idle minutes) "
                                    "to config.local.toml. Use it by name, e.g. voicectl say --model NAME.")
    q.add_argument("name", help="short name to request it by, e.g. parakeet")
    q.add_argument("repo", help="Hugging Face repo, e.g. mlx-community/parakeet-tdt-0.6b-v3")
    q.add_argument("--kind", choices=["stt", "tts"], help="default: from the repo's task on Hugging Face")
    q.add_argument("--languages", help="TTS: comma-separated languages it should speak, e.g. en,fr,de")
    q.add_argument("--voice", help="TTS: the model's speaker name to use by default, if it has named voices")
    q.add_argument("--preload", action="store_true", help="load at start and keep loaded (uses memory all the time)")
    q.add_argument("--extra-repo", action="append", help="helper repo the model loads by name (repeatable)")
    q = msub.add_parser("remove", parents=[common], help="remove a model added with `models add`, and its files")
    q.add_argument("name")
    q.add_argument("--keep-files", action="store_true", help="keep the downloaded files")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    q = msub.add_parser("delete", parents=[common], help="delete one downloaded model that is not in use")
    q.add_argument("repo")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    q = msub.add_parser("clean", parents=[common], help="delete every downloaded model that is not in use")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    p = sub.add_parser("bench", parents=[common], help="benchmark models: stt, tts, memory, report",
                       description="Benchmarks run each model in its own process, aborting it above 12 GB, and "
                                   "refuse to start when memory is short. Results go to bench/results/.")
    bsub = p.add_subparsers(dest="bench_cmd", required=True, metavar="what")
    q = bsub.add_parser("stt", parents=[common], help="accuracy (EN/FR/DE), speed and memory of STT models")
    q.add_argument("repos", nargs="*", help="Hugging Face repos (default: the configured STT models)")
    q = bsub.add_parser("tts", parents=[common], help="speed, memory, intelligibility and listening samples of a TTS model")
    q.add_argument("repo")
    q.add_argument("--voice", help="the model's speaker name, if it has named voices")
    q.add_argument("--languages", default="en,fr", help="comma-separated (default en,fr)")
    q = bsub.add_parser("memory", parents=[common], help="memory of the whole service (default config) under load")
    q.add_argument("--llm-gb", type=float, action="append", default=[], help="add an LLM's peak to the budget (repeatable)")
    bsub.add_parser("report", parents=[common], help="print the result tables")
    for q in bsub.choices.values():
        q.add_argument("--min-free-gb", type=float, default=14, help="refuse to start below this much free memory (default 14)")
        q.add_argument("--force", action="store_true", help="start even when memory is short")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("tokens", parents=[common], help="manage consumer tokens (one per program)",
                       description="Tokens are stored hashed; a new token is shown once. Changes apply at once.")
    tsub = p.add_subparsers(dest="tokens_cmd", required=True, metavar="action")
    tsub.add_parser("add", parents=[common], help="create a token for a consumer, print it once").add_argument("name")
    tsub.add_parser("list", parents=[common], help="consumer names and creation dates (never the tokens)")
    tsub.add_parser("revoke", parents=[common], help="delete a consumer's token").add_argument("name")
    p.set_defaults(func=cmd_tokens)

    p = sub.add_parser("config", parents=[common], help="show the effective configuration")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("selftest", parents=[common], help="live round trip (speak, then transcribe) in every language")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("update", parents=[common], help="update packages and models, test, roll back on failure",
                       description="Move Python packages (uv.lock) and models (models.lock.json) to their newest "
                                   "versions released at least --cooldown-days ago, run the tests, restart, "
                                   "self-test; restore the previous versions if anything fails.")
    p.add_argument("--check", action="store_true", help="only show what would change")
    p.add_argument("--cooldown-days", type=float, default=7, help="ignore releases younger than this (default 7)")
    p.add_argument("--no-restart", action="store_true", help="update files but do not restart or self-test")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("autoupdate", parents=[common], help="weekly automatic update: enable, disable, status")
    p.add_argument("action", choices=["enable", "disable", "status"])
    p.add_argument("--weekday", default="sun", choices=["sun", "mon", "tue", "wed", "thu", "fri", "sat"],
                   help="day of the week (default sun)")
    p.add_argument("--time", default="02:30", help="HH:MM, local time (default 02:30)")
    p.add_argument("--cooldown-days", type=float, default=7, help="ignore releases younger than this (default 7)")
    p.set_defaults(func=cmd_autoupdate)

    for name, text in (("start", "start the launchd agent"), ("stop", "stop it"),
                       ("restart", "restart it (after a config change)"),
                       ("check", "health 200 and 401 without a token; exit code for monitoring")):
        sub.add_parser(name, parents=[common], help=text).set_defaults(func=cmd_script)
    p = sub.add_parser("logs", parents=[common], help="follow the service log (Ctrl-C to stop)")
    p.add_argument("-n", "--lines", type=int, default=50)
    p.set_defaults(func=cmd_script)
    p = sub.add_parser("install", parents=[common], help="set up the environment and launchd agent (scripts/install.sh)")
    p.add_argument("--no-start", action="store_true")
    p.set_defaults(func=cmd_script)
    p = sub.add_parser("upgrade", parents=[common], help="apply uv.lock, fetch models, restart (scripts/upgrade.sh)")
    p.add_argument("--check", action="store_true", help="only list newer versions of the pinned packages")
    p.set_defaults(func=cmd_script)
    p = sub.add_parser("uninstall", parents=[common], help="remove the launchd agent (scripts/uninstall.sh)")
    p.add_argument("--purge", action="store_true", help="also remove environment, logs, tokens and models")
    p.set_defaults(func=cmd_script)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
