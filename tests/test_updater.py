"""Update rules, with PyPI and Hugging Face mocked: cooling-off, no downgrades, model pins."""

import datetime as dt
from types import SimpleNamespace

from voiceserver import modellock, updater

NOW = dt.datetime.now(dt.timezone.utc)


def days_ago(n):
    return NOW - dt.timedelta(days=n)


def test_diff_lists_changes_only():
    d = updater._diff({"a": "1", "b": "1", "c": "1"}, {"a": "1", "b": "2", "d": "1"})
    assert d == [{"name": "b", "old": "1", "new": "2"}, {"name": "c", "old": "1", "new": None},
                 {"name": "d", "old": None, "new": "1"}]


def test_plan_holds_back_young_releases(monkeypatch, cfg, tmp_path):
    (tmp_path / "uv.lock").write_text('name = "old-enough"\nversion = "1.0"\nname = "too-new"\nversion = "1.0"\n')
    (tmp_path / "pyproject.toml").write_text("")
    u = updater.Updater.__new__(updater.Updater)
    u.cooldown_days, u.lines, u._log = 7, [], lambda m: None
    monkeypatch.setattr(u, "uv", lambda: "uv")

    def fake_run(*cmd, cwd=None, **kw):  # "uv lock --upgrade" in the scratch copy
        from pathlib import Path
        Path(cwd, "uv.lock").write_text('name = "old-enough"\nversion = "2.0"\nname = "too-new"\nversion = "2.0"\n')
        return ""
    monkeypatch.setattr(u, "run", fake_run)
    released = {("old-enough", "2.0"): days_ago(30), ("too-new", "2.0"): days_ago(2)}
    monkeypatch.setattr(updater, "_pypi_release_date", lambda n, v: released[(n, v)])
    settled, young = u.plan_packages(tmp_path)
    assert [c["name"] for c in settled] == ["old-enough"]
    assert [c["name"] for c in young] == ["too-new"] and young[0]["old"] == "1.0"  # stays, no downgrade
    # PyPI unreachable: an unknown release date is never treated as settled.
    monkeypatch.setattr(updater, "_pypi_release_date", lambda n, v: None)
    settled, young = u.plan_packages(tmp_path)
    assert settled == [] and {c["name"] for c in young} == {"old-enough", "too-new"}
    assert young[0]["released"] == "unknown"


def test_model_pin_takes_newest_settled_commit(monkeypatch):
    commits = [SimpleNamespace(commit_id="new", created_at=days_ago(1)),
               SimpleNamespace(commit_id="settled", created_at=days_ago(10)),
               SimpleNamespace(commit_id="old", created_at=days_ago(100))]
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(list_repo_commits=lambda repo: commits))
    assert modellock._newest_settled_commit("x/y", 7)[0] == "settled"
    assert modellock._newest_settled_commit("x/y", 0.5)[0] == "new"
    assert modellock._newest_settled_commit("x/y", 1000)[0] == "old"  # repo younger than cooling-off


def test_required_repos_include_helpers(cfg):
    cfg.models["quality"].extra_repos = ["fake/tokenizer"]
    repos = modellock.required_repos(cfg)
    assert "fake/tokenizer" in repos and "fake/whisper-large-v3-turbo" in repos and len(repos) == len(set(repos))


def test_schedule_plist(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "UPDATE_PLIST", tmp_path / "update.plist")
    monkeypatch.setattr(updater.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert updater.enable_schedule("sunday", "02:30", 7, tmp_path) == "weekly on Sun at 02:30"
    assert updater.schedule_description() == "weekly on Sun at 02:30, cooling-off 7 days"
    assert updater.disable_schedule() and updater.schedule_description() is None
