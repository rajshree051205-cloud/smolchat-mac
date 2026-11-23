"""Tests for the single-source app version (Issue #19)."""

from __future__ import annotations

import ast
from pathlib import Path

from app import config
from app.config import APP_VERSION, read_project_version

ROOT = Path(__file__).resolve().parent.parent


def test_app_version_matches_pyproject():
    assert read_project_version(ROOT / "pyproject.toml") == APP_VERSION
    assert APP_VERSION != config.FALLBACK_VERSION


def test_reads_only_the_project_table(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[build-system]\nversion = "9.9.9"\n\n'
        '[project]\nname = "smolchat"\nversion = "1.2.3"\n\n'
        '[tool.other]\nversion = "8.8.8"\n',
        encoding="utf-8",
    )
    assert read_project_version(pyproject) == "1.2.3"


def test_missing_file_or_version_returns_none(tmp_path):
    assert read_project_version(tmp_path / "missing.toml") is None
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "smolchat"\n', encoding="utf-8")
    assert read_project_version(pyproject) is None


def test_frozen_build_reads_bundled_pyproject(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "4.5.6"\n', encoding="utf-8")
    monkeypatch.setattr(config.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert config._app_version() == "4.5.6"


def test_version_is_not_hard_coded_elsewhere():
    spec = (ROOT / "packaging" / "SmolChat.spec").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(spec)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "CFBundleShortVersionString":
                    assert not isinstance(value, ast.Constant)
    assert '"pyproject.toml"' in spec
