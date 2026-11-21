"""Tests for configuration and Settings validation (Issue #4)."""

import json
from pathlib import Path

import pytest

from app.config import (
    ALLOWED_ACCENTS,
    ALLOWED_ROUTING_PREFERENCES,
    ALLOWED_THEMES,
    Settings,
)


class TestSettingsValidation:
    """Test field-level validation, type coercion, and defaults in Settings."""

    def test_default_settings(self):
        s = Settings()
        assert s.theme == "dark"
        assert s.accent == "violet"
        assert s.routing_preference == "balanced"
        assert s.default_model_id == ""
        assert s.temperature == 0.7
        assert s.max_tokens == 1024
        assert s.context_length == 4096
        assert s.use_gpu is True
        assert s.enable_thinking is False
        assert "SmolChat" in s.system_prompt

    def test_validate_allowed_options(self):
        # Valid themes
        for theme in ALLOWED_THEMES:
            assert Settings.validate_value("theme", theme) == theme

        with pytest.raises(ValueError):
            Settings.validate_value("theme", "purple")

        # Valid accents
        for accent in ALLOWED_ACCENTS:
            assert Settings.validate_value("accent", accent) == accent

        with pytest.raises(ValueError):
            Settings.validate_value("accent", "crimson")

        # Valid routing preferences
        for pref in ALLOWED_ROUTING_PREFERENCES:
            assert Settings.validate_value("routing_preference", pref) == pref

        with pytest.raises(ValueError):
            Settings.validate_value("routing_preference", "fast")

    def test_validate_numeric_ranges(self):
        # Temperature
        assert Settings.validate_value("temperature", 0.0) == 0.0
        assert Settings.validate_value("temperature", 1.5) == 1.5
        assert Settings.validate_value("temperature", "0.8") == 0.8
        with pytest.raises(ValueError):
            Settings.validate_value("temperature", -0.1)
        with pytest.raises(ValueError):
            Settings.validate_value("temperature", 2.1)
        with pytest.raises(TypeError):
            Settings.validate_value("temperature", "hot")
        with pytest.raises(TypeError):
            Settings.validate_value("temperature", True)

        # max_tokens
        assert Settings.validate_value("max_tokens", 512) == 512
        assert Settings.validate_value("max_tokens", "2048") == 2048
        with pytest.raises(ValueError):
            Settings.validate_value("max_tokens", 0)
        with pytest.raises(ValueError):
            Settings.validate_value("max_tokens", -10)
        with pytest.raises(ValueError):
            Settings.validate_value("max_tokens", 99999)
        with pytest.raises(TypeError):
            Settings.validate_value("max_tokens", "many")
        with pytest.raises(TypeError):
            Settings.validate_value("max_tokens", True)

        # context_length
        assert Settings.validate_value("context_length", 2048) == 2048
        assert Settings.validate_value("context_length", "8192") == 8192
        with pytest.raises(ValueError):
            Settings.validate_value("context_length", 256)
        with pytest.raises(ValueError):
            Settings.validate_value("context_length", 65536)
        with pytest.raises(TypeError):
            Settings.validate_value("context_length", "huge")
        with pytest.raises(TypeError):
            Settings.validate_value("context_length", False)

    def test_validate_booleans_and_strings(self):
        assert Settings.validate_value("use_gpu", True) is True
        assert Settings.validate_value("use_gpu", False) is False
        with pytest.raises(TypeError):
            Settings.validate_value("use_gpu", "yes")

        assert Settings.validate_value("enable_thinking", True) is True
        assert Settings.validate_value("enable_thinking", False) is False
        with pytest.raises(TypeError):
            Settings.validate_value("enable_thinking", 1)

        assert Settings.validate_value("default_model_id", "smollm2-360m") == "smollm2-360m"
        with pytest.raises(TypeError):
            Settings.validate_value("default_model_id", 123)

        assert Settings.validate_value("system_prompt", "Custom prompt") == "Custom prompt"
        with pytest.raises(TypeError):
            Settings.validate_value("system_prompt", None)

    def test_load_from_corrupted_or_out_of_range_json(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        raw_data = {
            "theme": "purple",             # invalid option -> default 'dark'
            "accent": "neon",              # invalid option -> default 'violet'
            "temperature": "hot",          # invalid type -> default 0.7
            "max_tokens": -5,              # out of range -> default 1024
            "context_length": 999999,      # out of range -> default 4096
            "routing_preference": "fast",  # invalid option -> default 'balanced'
            "use_gpu": "maybe",            # invalid type -> default True
            "system_prompt": "Keep this valid prompt",  # VALID -> preserved
        }
        settings_file.write_text(json.dumps(raw_data), encoding="utf-8")

        loaded = Settings.load(settings_file)
        # Invalid fields must fall back to their defaults
        assert loaded.theme == "dark"
        assert loaded.accent == "violet"
        assert loaded.temperature == 0.7
        assert loaded.max_tokens == 1024
        assert loaded.context_length == 4096
        assert loaded.routing_preference == "balanced"
        assert loaded.use_gpu is True

        # Valid field must be preserved!
        assert loaded.system_prompt == "Keep this valid prompt"

    def test_load_nonexistent_or_malformed_json(self, tmp_path: Path):
        # Nonexistent file returns defaults
        missing = tmp_path / "nonexistent.json"
        assert Settings.load(missing) == Settings()

        # Malformed json returns defaults
        malformed = tmp_path / "broken.json"
        malformed.write_text("{not valid json", encoding="utf-8")
        assert Settings.load(malformed) == Settings()

        # Non-dict json returns defaults
        array_file = tmp_path / "array.json"
        array_file.write_text("[1, 2, 3]", encoding="utf-8")
        assert Settings.load(array_file) == Settings()
