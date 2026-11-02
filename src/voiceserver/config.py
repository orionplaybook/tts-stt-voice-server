"""Configuration: config.toml in the project folder, optionally overlaid by config.local.toml
(machine-specific settings such as the token file path; never committed to a public copy).
VOICE_SERVER_CONFIG points at another base file. No secrets live in either file."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engines import VoiceSpec

PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    tokens_file: Path = Path("~/.config/voice-server/tokens.json")
    log_dir: Path = Path("~/Library/Logs/voice-server")
    log_content: bool = False          # debug only: log transcripts and TTS input
    max_upload_mb: float = 25
    max_audio_minutes: float = 30      # longest audio accepted (decoded length), uploads and realtime
    max_input_chars: int = 4096
    queue_timeout_s: float = 30        # waiting for a free slot
    request_timeout_s: float = 180     # one inference job
    mlx_cache_limit_gb: float = 0.25   # MLX keeps freed GPU buffers up to this much
    release_after_request: bool = True # give freed GPU buffers back after every job
    tls_cert: str = ""                 # PEM certificate and key: serve HTTPS (required off loopback)
    tls_key: str = ""
    allow_plain_http_remote: bool = False  # explicit opt-in to plain HTTP on a non-loopback address


@dataclass
class ModelConfig:
    name: str
    kind: str                          # "stt" or "tts"
    repo: str
    preload: bool = True
    idle_unload_s: float = 0           # 0 = keep loaded
    max_queue: int = 8                 # requests waiting or running for this model
    languages: list[str] = field(default_factory=list)
    default_voice: str = ""
    voices: dict[str, dict[str, VoiceSpec]] = field(default_factory=dict)  # voice -> lang|"*" -> spec
    fallback: str = ""                 # TTS model used for languages this one does not speak
    extra_repos: list[str] = field(default_factory=list)  # helper repos the model loads by name
    notes: str = ""


@dataclass
class Config:
    server: ServerConfig
    stt_default: str
    tts_default: str
    models: dict[str, ModelConfig]
    aliases: dict[str, str]            # alias -> model name (both kinds)
    voice_aliases: dict[str, str]      # e.g. OpenAI "nova" -> configured voice name
    source: list[Path]

    def resolve(self, name: str | None, kind: str) -> ModelConfig | None:
        if not name:
            name = self.stt_default if kind == "stt" else self.tts_default
        name = self.aliases.get(name, name)
        model = self.models.get(name)
        if model is None:  # also accept the Hugging Face repo id itself, if configured
            model = next((m for m in self.models.values() if m.repo == name), None)
        return model if model and model.kind == kind else None


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def is_loopback(host: str) -> bool:
    return host in LOOPBACK or host.startswith("127.")


def base_url(server: "ServerConfig") -> str:
    """URL a local client uses to reach this server (loopback when it listens everywhere)."""
    scheme = "https" if server.tls_cert and server.tls_key else "http"
    host = "127.0.0.1" if server.host in ("0.0.0.0", "::", "") else server.host
    return f"{scheme}://{host}:{server.port}"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _voice_table(raw: dict[str, Any], model_dir: Path) -> dict[str, dict[str, VoiceSpec]]:
    voices: dict[str, dict[str, VoiceSpec]] = {}
    for vname, spec in raw.items():
        per_lang = spec if any(isinstance(v, dict) for v in spec.values()) else {"*": spec}
        voices[vname] = {}
        for lang, s in per_lang.items():
            ref = s.get("ref_audio")
            ref_path = str((model_dir / ref).resolve()) if ref else None
            ref_text = s.get("ref_text")
            if not ref_text and s.get("ref_text_file"):
                ref_text = (model_dir / s["ref_text_file"]).read_text().strip()
            voices[vname][lang] = VoiceSpec(speaker=s.get("speaker"), ref_audio=ref_path,
                                            ref_text=ref_text, instruct=s.get("instruct"))
    return voices


def load_config(path: str | os.PathLike | None = None) -> Config:
    base = Path(path or os.environ.get("VOICE_SERVER_CONFIG") or PROJECT_DIR / "config.toml")
    data = tomllib.loads(base.read_text())
    sources = [base]
    local = base.with_name("config.local.toml")
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text()))
        sources.append(local)

    s = data.get("server", {})
    server = ServerConfig(**{k: v for k, v in s.items() if k in ServerConfig.__dataclass_fields__})
    server.tokens_file = Path(str(server.tokens_file)).expanduser()
    server.log_dir = Path(str(server.log_dir)).expanduser()
    for attr in ("tls_cert", "tls_key"):
        if getattr(server, attr):
            setattr(server, attr, str(Path(getattr(server, attr)).expanduser()))

    models: dict[str, ModelConfig] = {}
    aliases: dict[str, str] = {}
    for kind in ("stt", "tts"):
        section = data.get(kind, {})
        for name, m in section.get("models", {}).items():
            if name in models:
                raise ValueError(f"model name {name!r} is used for both STT and TTS")
            models[name] = ModelConfig(
                name=name, kind=kind, repo=m["repo"], preload=m.get("preload", True),
                idle_unload_s=float(m.get("idle_unload_s", 0)), max_queue=int(m.get("max_queue", 8)),
                languages=list(m.get("languages", [])), default_voice=m.get("default_voice", ""),
                voices=_voice_table(m.get("voices", {}), base.parent), notes=m.get("notes", ""),
                fallback=m.get("fallback", ""), extra_repos=list(m.get("extra_repos", [])))
        aliases.update(section.get("aliases", {}))
    stt_default = data.get("stt", {}).get("default", "")
    tts_default = data.get("tts", {}).get("default", "")
    for alias, target in aliases.items():
        if target not in models:
            raise ValueError(f"alias {alias!r} points at unknown model {target!r}")
    for m in models.values():
        if m.fallback and (m.fallback not in models or models[m.fallback].kind != m.kind or m.fallback == m.name):
            raise ValueError(f"model {m.name!r}: fallback {m.fallback!r} is not another configured {m.kind} model")
    for kind, default in (("stt", stt_default), ("tts", tts_default)):
        if default not in models or models[default].kind != kind:
            raise ValueError(f"[{kind}] default {default!r} is not a configured {kind} model")
    return Config(server=server, stt_default=stt_default, tts_default=tts_default, models=models,
                  aliases=aliases, voice_aliases=dict(data.get("tts", {}).get("voice_aliases", {})),
                  source=sources)
