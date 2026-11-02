"""Update everything the service runs on, safely: Python packages (uv.lock) and models
(models.lock.json), with a cooling-off period, tests, a live self-test, and automatic rollback.

  voicectl update            check for updates and apply them
  voicectl update --check    only report what would change
  voicectl autoupdate enable [--weekday sun --time 02:30]   run `update` weekly via launchd

Steps of an update:
  1. Back up pyproject.toml, uv.lock and models.lock.json.
  2. Move each package in uv.lock to its newest release, but only if that release is at least
     `cooldown_days` old (checked on PyPI; younger ones wait, nothing moves backwards), and each
     model pin to its newest commit at least that old.
  3. Nothing changed: stop. Otherwise install the new environment and models.
  4. Run the unit tests, restart the service, run `voicectl selftest` against it.
  5. Any failure: restore the backups, reinstall the previous environment, restart, self-test.

Every run writes update-status.json (for monitoring) and appends to update.log, both in the
service's log folder. Not updated here: ffmpeg (installed by you, e.g. Homebrew) and uv itself
(pinned in scripts/common.sh); the status reports their versions.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import PROJECT_DIR, load_config

LOCKED_FILES = ("pyproject.toml", "uv.lock", "models.lock.json")
UPDATE_LABEL = "local.voice-server.update"
UPDATE_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{UPDATE_LABEL}.plist"


class StepFailed(Exception):
    pass


class Updater:
    def __init__(self, config: str | None = None, cooldown_days: float = 7, log=print):
        self.config = config
        self.cfg = load_config(config)
        self.cooldown_days = cooldown_days
        self.log_dir = self.cfg.server.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = log
        self.lines: list[str] = []

    # --- plumbing ------------------------------------------------------------------------

    def say(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append(f"{stamp} {msg}")
        self._log(msg)

    def run(self, *cmd: str, env: dict | None = None, timeout: float = 1800, cwd=None) -> str:
        e = {**os.environ, **_uv_env(), **(env or {})}
        e.pop("HF_HUB_OFFLINE", None)  # updates need the network
        p = subprocess.run(cmd, cwd=cwd or PROJECT_DIR, capture_output=True, text=True, env=e, timeout=timeout)
        if p.returncode != 0:
            raise StepFailed(f"{' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))} failed:\n"
                             + (p.stderr or p.stdout)[-1500:])
        return p.stdout

    def uv(self) -> str:
        return subprocess.run(["/bin/bash", "-c", f"source '{PROJECT_DIR}/scripts/common.sh' && uv_bin"],
                              capture_output=True, text=True, check=True).stdout.strip()

    # --- the update ----------------------------------------------------------------------

    def packages(self) -> dict[str, str]:
        lock = (PROJECT_DIR / "uv.lock").read_text()
        out, name = {}, None
        for line in lock.splitlines():
            if line.startswith("name = "):
                name = line.split('"')[1]
            elif line.startswith("version = ") and name:
                out[name] = line.split('"')[1]
                name = None
        return out

    def plan_packages(self, workdir: Path) -> tuple[list[dict], list[dict]]:
        """Resolve the newest versions in a scratch copy of the project, then split the changes
        into settled ones (released at least cooldown_days ago) and too-young ones. Never plans a
        downgrade: a package whose newest version is too young simply stays where it is."""
        with tempfile.TemporaryDirectory() as tmp:
            for f in ("pyproject.toml", "uv.lock"):
                shutil.copy2(workdir / f, Path(tmp) / f)
            (Path(tmp) / "README.md").write_text("x")
            self.run(self.uv(), "lock", "--upgrade", "--quiet", cwd=tmp)
            latest = _packages_in(Path(tmp) / "uv.lock")
        settled, young = [], []
        for change in _diff(_packages_in(workdir / "uv.lock"), latest):
            if change["new"] is None:
                settled.append(change)  # no longer needed
                continue
            released = _pypi_release_date(change["name"], change["new"])
            change["released"] = released.strftime("%Y-%m-%d") if released else "unknown"
            # An unknown release date (PyPI unreachable, or not a PyPI package) never counts as
            # settled: the cooling-off period is a safety control, so in doubt the package waits.
            if released is not None and released <= self.cutoff_dt():
                settled.append(change)
            else:
                young.append(change)
        return settled, young

    def lock_settled(self, workdir: Path, settled: list[dict]) -> None:
        """Re-lock allowing only the settled packages to move."""
        names = [c["name"] for c in settled if c["new"]]
        if not names:
            return
        args = [a for n in names for a in ("--upgrade-package", n)]
        self.run(self.uv(), "lock", *args, "--quiet", cwd=workdir)

    def check(self) -> dict:
        """What an update would change, without changing anything."""
        from . import modellock
        settled, young = self.plan_packages(PROJECT_DIR)
        model_changes = []
        for repo in modellock.required_repos(self.cfg):
            cur = modellock.revision(repo)
            new, date = modellock._newest_settled_commit(repo, self.cooldown_days)
            if cur != new:
                model_changes.append({"repo": repo, "old": cur, "new": new, "date": date})
        return {"packages": settled, "waiting": young, "models": model_changes}

    def cutoff_dt(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=self.cooldown_days)

    def apply(self, restart: bool = True) -> dict:
        from . import modellock
        started = time.time()
        backup = self.log_dir / "update-backup"
        shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir(parents=True)
        for f in LOCKED_FILES:
            if (PROJECT_DIR / f).exists():
                shutil.copy2(PROJECT_DIR / f, backup / f)
        status = {"started": _now(), "cooldown_days": self.cooldown_days, "result": None,
                  "packages": [], "models": [], "error": None}
        uv = self.uv()
        try:
            before = self.packages()
            self.say(f"checking packages (only releases at least {self.cooldown_days:g} days old)")
            settled, young = self.plan_packages(PROJECT_DIR)
            status["waiting"] = young
            for y in young:
                self.say(f"waiting: {y['name']} {y['new']} (released {y['released']}: too recent or unknown)")
            self.lock_settled(PROJECT_DIR, settled)
            status["packages"] = _diff(before, self.packages())
            self.say("checking models")
            status["models"] = modellock.sync(self.cfg, update=True, cooldown_days=self.cooldown_days, quiet=True)
            if not status["packages"] and not status["models"]:
                status["result"] = "up-to-date"
                self.say("everything is up to date")
                return self._finish(status, started)
            for p in status["packages"]:
                self.say(f"package {p['name']}: {p['old'] or '-'} -> {p['new'] or 'removed'}")
            for m in status["models"]:
                self.say(f"model {m['repo']}: {(m['old'] or '-')[:10]} -> {(m['new'] or 'removed')[:10]} ({m['date']})")
            self.say("installing")
            self.run(uv, "sync", "--frozen", "--all-groups", "--quiet")
            self.say("running unit tests")
            self.run(sys.executable, "-m", "pytest", "-q", "-x", "tests", "--ignore=tests/e2e", timeout=900)
            if restart:
                self.restart_and_selftest()
            if os.environ.get("VOICE_SERVER_UPDATE_SIMULATE_FAILURE"):  # to test the rollback
                raise StepFailed("simulated failure (VOICE_SERVER_UPDATE_SIMULATE_FAILURE)")
            status["result"] = "updated"
            self.say("update complete")
        except (StepFailed, subprocess.TimeoutExpired, Exception) as e:  # noqa: BLE001 - always roll back
            status["error"] = str(e)[-2000:]
            self.say(f"FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}; rolling back")
            try:
                for f in LOCKED_FILES:
                    if (backup / f).exists():
                        shutil.copy2(backup / f, PROJECT_DIR / f)
                self.run(uv, "sync", "--frozen", "--all-groups", "--quiet")
                modellock.pin_main_refs()
                if restart:
                    self.restart_and_selftest()
                status["result"] = "rolled-back"
                self.say("rolled back to the previous versions; service is healthy")
            except Exception as e2:  # noqa: BLE001
                status["result"] = "failed"
                status["error"] += f"\nrollback also failed: {e2}"[-1500:]
                self.say(f"ROLLBACK FAILED: {e2}; see {self.log_dir / 'update.log'}")
        return self._finish(status, started)

    def restart_and_selftest(self) -> None:
        self.say("restarting the service")
        self.run("/bin/bash", str(PROJECT_DIR / "scripts" / "voice-server.sh"), "restart", timeout=600)
        self.say("self-test")
        out = self.run(str(PROJECT_DIR / ".venv" / "bin" / "python"), "-m", "voiceserver.cli", "selftest",
                       *(["--config", self.config] if self.config else []), timeout=600)
        for line in out.strip().splitlines():
            self.say("  " + line)

    def _finish(self, status: dict, started: float) -> dict:
        status["finished"] = _now()
        status["duration_s"] = round(time.time() - started)
        status["git_uncommitted"] = _git_changed()
        status["tools"] = _tool_versions()
        (self.log_dir / "update-status.json").write_text(json.dumps(status, indent=1) + "\n")
        with open(self.log_dir / "update.log", "a") as f:
            f.write("\n".join(self.lines) + "\n")
        _trim(self.log_dir / "update.log", 2000)
        return status


# --- helpers ----------------------------------------------------------------------------------

def _uv_env() -> dict[str, str]:
    return {"UV_PYTHON_INSTALL_DIR": str(PROJECT_DIR / ".python"), "UV_CACHE_DIR": str(PROJECT_DIR / ".uv-cache"),
            "UV_PYTHON_PREFERENCE": "only-managed"}


def _packages_in(lock: Path) -> dict[str, str]:
    out, name = {}, None
    for line in lock.read_text().splitlines():
        if line.startswith("name = "):
            name = line.split('"')[1]
        elif line.startswith("version = ") and name:
            out[name] = line.split('"')[1]
            name = None
    return out


def _diff(before: dict[str, str], after: dict[str, str]) -> list[dict]:
    names = sorted(set(before) | set(after))
    return [{"name": n, "old": before.get(n), "new": after.get(n)} for n in names if before.get(n) != after.get(n)]


def _pypi_release_date(name: str, version: str) -> dt.datetime | None:
    """Upload time of that release on PyPI (None if not a PyPI package, e.g. a direct URL)."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=20) as r:
            files = json.load(r).get("urls", [])
        times = [f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601")]
        return min(dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times) if times else None
    except Exception:
        return None


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_changed() -> list[str]:
    p = subprocess.run(["git", "status", "--porcelain", "--", *LOCKED_FILES], cwd=PROJECT_DIR,
                       capture_output=True, text=True)
    return [line[3:] for line in p.stdout.splitlines()] if p.returncode == 0 else []


def _tool_versions() -> dict[str, str]:
    def first_line(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        except Exception:
            return "not found"
    return {"ffmpeg": first_line("ffmpeg", "-version"), "python": sys.version.split()[0]}


def _trim(path: Path, keep_lines: int) -> None:
    lines = path.read_text().splitlines()
    if len(lines) > keep_lines:
        path.write_text("\n".join(lines[-keep_lines:]) + "\n")


# --- schedule ---------------------------------------------------------------------------------

WEEKDAYS = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}


def enable_schedule(weekday: str, at: str, cooldown_days: float, log_dir: Path) -> str:
    hour, minute = (int(x) for x in at.split(":"))
    day = WEEKDAYS[weekday[:3].lower()]
    UPDATE_PLIST.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{UPDATE_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{PROJECT_DIR}/voicectl</string><string>update</string>
    <string>--cooldown-days</string><string>{cooldown_days:g}</string>
  </array>
  <key>WorkingDirectory</key><string>{PROJECT_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>{day}</integer><key>Hour</key><integer>{hour}</integer>
        <key>Minute</key><integer>{minute}</integer></dict>
  <key>StandardOutPath</key><string>{log_dir}/update-launchd.log</string>
  <key>StandardErrorPath</key><string>{log_dir}/update-launchd.log</string>
</dict>
</plist>
""")
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{UPDATE_LABEL}"], capture_output=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(UPDATE_PLIST)], check=True)
    return f"weekly on {weekday[:3].capitalize()} at {hour:02d}:{minute:02d}"


def disable_schedule() -> bool:
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{UPDATE_LABEL}"], capture_output=True)
    existed = UPDATE_PLIST.exists()
    UPDATE_PLIST.unlink(missing_ok=True)
    return existed


def schedule_description() -> str | None:
    if not UPDATE_PLIST.exists():
        return None
    import plistlib
    p = plistlib.loads(UPDATE_PLIST.read_bytes())
    c = p["StartCalendarInterval"]
    day = next(k for k, v in WEEKDAYS.items() if v == c["Weekday"])
    cd = p["ProgramArguments"][-1]
    return f"weekly on {day.capitalize()} at {c['Hour']:02d}:{c['Minute']:02d}, cooling-off {cd} days"
