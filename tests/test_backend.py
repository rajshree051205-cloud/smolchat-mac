"""Tests for app/backend.py.

Covers:
- Helper functions: _date_section, Backend._build_prompt, Backend._progress_text.
- DictListModel behaviors.
- Backend conversation lifecycle and messages.
- Generation lifecycle: sendMessage, stopGeneration, regenerate.
- Conversation deletion and attachment file cleanup.
- Model state, download blockers, and model deletion.
- Manual-mode deleted-model error handling.
- Settings get/set type coercion, persistence, and model unloading.
"""

from __future__ import annotations
import threading
import datetime
import time
from unittest.mock import MagicMock

import pytest
from PySide6.QtGui import QGuiApplication

from app.backend import Backend, DictListModel, _date_section
from app.config import HardwareInfo, Settings
from app.database import Message
from app.inference import GenerationResult, InferenceEngine, LoadSpec
from app.model_manager import (
    DownloadCancelled,
    InstalledModel,
    ModelEntry,
    ModelFile,
    ModelRegistry,
    ModelStore,
)





def create_test_model_entry(
    model_id: str = "test-model",
    name: str = "Test Model",
    family: str = "SmolLM2",
    capabilities: tuple[str, ...] = ("chat",),
    files: tuple[ModelFile, ...] | None = None,
) -> ModelEntry:
    if files is None:
        files = (
            ModelFile(
                role="model",
                filename="model.gguf",
                size=1024,
                sha256="a" * 64,
            ),
        )
    return ModelEntry.from_dict(
        {
            "id": model_id,
            "name": name,
            "family": family,
            "parameters": "135M",
            "params_b": 0.135,
            "quality": 0.135,
            "description": "A lightweight model for unit testing",
            "capabilities": list(capabilities),
            "backend": "llama_cpp",
            "repo": "smollabs/test-model",
            "revision": "a" * 40,
            "source": "Hugging Face",
            "license": "Apache-2.0",
            "quantization": "Q4_K_M",
            "context_length": 2048,
            "recommended_ram_gb": 8,
            "files": [
                {
                    "role": f.role,
                    "filename": f.filename,
                    "size": f.size,
                    "sha256": f.sha256,
                }
                for f in files
            ],
        }
    )


def make_test_message(
    id: int,
    role: str,
    content: str,
    metadata: dict | None = None,
    conversation_id: str = "c1",
) -> Message:
    return Message(
        id=id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=1000.0 + id,
        metadata=metadata or {},
    )


# =====================================================================
# Phase 4 — Pure Helper Tests
# =====================================================================


class TestDateSection:
    """Test date grouping boundaries for _date_section."""

    def test_today(self):
        fixed_now = datetime.datetime(2026, 11, 18, 14, 30, 0)
        ts = datetime.datetime(2026, 11, 18, 8, 0, 0).timestamp()
        assert _date_section(ts, now=fixed_now) == "Today"

    def test_yesterday(self):
        fixed_now = datetime.datetime(2026, 11, 18, 14, 30, 0)
        ts = datetime.datetime(2026, 11, 17, 23, 59, 59).timestamp()
        assert _date_section(ts, now=fixed_now) == "Yesterday"

    def test_previous_7_days_boundaries(self):
        fixed_now = datetime.datetime(2026, 11, 18, 12, 0, 0)
        # 2 days ago
        ts_2d = datetime.datetime(2026, 11, 16, 12, 0, 0).timestamp()
        assert _date_section(ts_2d, now=fixed_now) == "Previous 7 Days"
        # 6 days ago (today - 6 days)
        ts_6d = datetime.datetime(2026, 11, 12, 0, 0, 0).timestamp()
        assert _date_section(ts_6d, now=fixed_now) == "Previous 7 Days"

    def test_previous_30_days_boundaries(self):
        fixed_now = datetime.datetime(2026, 11, 18, 12, 0, 0)
        # Exactly 7 days ago (today - 7 days -> day <= today - 7 days, but > today - 30 days)
        ts_7d = datetime.datetime(2026, 11, 11, 12, 0, 0).timestamp()
        assert _date_section(ts_7d, now=fixed_now) == "Previous 30 Days"
        # 29 days ago
        ts_29d = datetime.datetime(2026, 10, 20, 12, 0, 0).timestamp()
        assert _date_section(ts_29d, now=fixed_now) == "Previous 30 Days"

    def test_month_grouping_same_year(self):
        fixed_now = datetime.datetime(2026, 11, 18, 12, 0, 0)
        # Earlier this year (e.g. May 2026, more than 30 days ago)
        ts_may = datetime.datetime(2026, 5, 10, 12, 0, 0).timestamp()
        assert _date_section(ts_may, now=fixed_now) == "May"

    def test_month_and_year_transition_different_year(self):
        fixed_now = datetime.datetime(2026, 1, 15, 12, 0, 0)
        # 45 days ago in previous year (November 2025)
        ts_prev_year = datetime.datetime(2025, 11, 1, 12, 0, 0).timestamp()
        assert _date_section(ts_prev_year, now=fixed_now) == "November 2025"

    def test_default_now_parameter(self):
        now_ts = datetime.datetime.now().timestamp()
        assert _date_section(now_ts) == "Today"


class TestProgressText:
    """Test Backend._progress_text helper."""

    def test_downloading_with_speed(self):
        text = Backend._progress_text(done=500_000_000, total=1_000_000_000, speed=50_000_000.0)
        assert "500 MB of 1.00 GB" in text
        assert "50%" in text
        assert "50.0 MB/s" in text
        assert "less than a minute left" in text

    def test_downloading_without_speed(self):
        text = Backend._progress_text(done=250_000_000, total=1_000_000_000, speed=0.0)
        assert "250 MB of 1.00 GB" in text
        assert "25%" in text
        assert "/s" not in text

    def test_boundary_values_zero_and_completed(self):
        # 0 bytes done
        text_0 = Backend._progress_text(done=0, total=1_000_000_000)
        assert "0 B of 1.00 GB" in text_0
        assert "0%" in text_0

        # 100% completed
        text_100 = Backend._progress_text(done=1_000_000_000, total=1_000_000_000)
        assert "1.00 GB of 1.00 GB" in text_100
        assert "100%" in text_100

    def test_zero_total_graceful_handling(self):
        text = Backend._progress_text(done=0, total=0)
        assert "0 B of 0 B" in text


class TestBuildPrompt:
    """Test Backend._build_prompt prompt compilation."""

    @pytest.fixture
    def backend(self, paths):
        b = Backend(paths)
        yield b
        b.shutdown()

    def test_system_prompt_handling(self, backend):
        backend.settings.system_prompt = "You are a helpful assistant."
        entry = create_test_model_entry(family="SmolLM2")
        history = [make_test_message(id=1, role="user", content="Hello")]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 2
        assert prompt[0].role == "system"
        assert prompt[0].content == "You are a helpful assistant."
        assert prompt[1].role == "user"
        assert prompt[1].content == "Hello"

    def test_smollm3_thinking_enabled(self, backend):
        backend.settings.system_prompt = "Be concise."
        backend.settings.enable_thinking = True
        entry = create_test_model_entry(family="SmolLM3")
        history = [make_test_message(id=1, role="user", content="Solve 2+2")]
        prompt = backend._build_prompt(history, entry)
        assert prompt[0].role == "system"
        assert prompt[0].content == "Be concise.\n/think"

    def test_smollm3_thinking_disabled(self, backend):
        backend.settings.system_prompt = "Be concise."
        backend.settings.enable_thinking = False
        entry = create_test_model_entry(family="SmolLM3")
        history = [make_test_message(id=1, role="user", content="Solve 2+2")]
        prompt = backend._build_prompt(history, entry)
        assert prompt[0].role == "system"
        assert prompt[0].content == "Be concise.\n/no_think"

    def test_system_prompt_excluded_for_vision_models(self, backend):
        backend.settings.system_prompt = "Be helpful."
        entry = create_test_model_entry(family="SmolVLM", capabilities=("chat", "vision"))
        history = [make_test_message(id=1, role="user", content="What is in this image?")]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 1
        assert prompt[0].role == "user"

    def test_skipping_error_messages(self, backend):
        backend.settings.system_prompt = ""
        entry = create_test_model_entry()
        history = [
            make_test_message(id=1, role="user", content="First question"),
            make_test_message(
                id=2, role="assistant", content="Error occurred", metadata={"error": True}
            ),
            make_test_message(id=3, role="user", content="Try again"),
        ]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 2
        assert [p.content for p in prompt] == ["First question", "Try again"]

    def test_skipping_empty_replies(self, backend):
        backend.settings.system_prompt = ""
        entry = create_test_model_entry()
        history = [
            make_test_message(id=1, role="user", content="Question"),
            make_test_message(id=2, role="assistant", content="   "),
            make_test_message(id=3, role="user", content="Next question"),
        ]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 2
        assert [p.content for p in prompt] == ["Question", "Next question"]

    def test_stripping_reasoning_from_assistant_turns(self, backend):
        backend.settings.system_prompt = ""
        entry = create_test_model_entry()
        history = [
            make_test_message(id=1, role="user", content="Question"),
            make_test_message(
                id=2, role="assistant", content="<think>internal thought</think>Direct answer"
            ),
            make_test_message(id=3, role="user", content="Follow-up"),
        ]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 3
        assert prompt[1].role == "assistant"
        assert prompt[1].content == "Direct answer"

    def test_images_retained_only_on_last_user_turn(self, backend, tmp_path):
        img1 = tmp_path / "img1.png"
        img1.write_bytes(b"data1")
        img2 = tmp_path / "img2.png"
        img2.write_bytes(b"data2")

        backend.settings.system_prompt = ""
        entry = create_test_model_entry(family="SmolVLM", capabilities=("chat", "vision"))
        history = [
            make_test_message(
                id=1, role="user", content="Look at 1", metadata={"images": [str(img1)]}
            ),
            make_test_message(id=2, role="assistant", content="I see it."),
            make_test_message(
                id=3, role="user", content="Look at 2", metadata={"images": [str(img2)]}
            ),
        ]
        prompt = backend._build_prompt(history, entry)
        assert len(prompt) == 3
        assert prompt[0].images == []
        assert prompt[2].images == [str(img2)]

    def test_empty_and_non_user_assistant_roles(self, backend):
        backend.settings.system_prompt = ""
        entry = create_test_model_entry()
        history = [
            make_test_message(id=1, role="tool", content="Tool output"),
            make_test_message(id=2, role="unknown", content="???"),
        ]
        prompt = backend._build_prompt(history, entry)
        assert prompt == []


# =====================================================================
# Phase 5 — Backend Behaviour Tests
# =====================================================================


class FakeTestEngine(InferenceEngine):
    """Mock engine that mimics InferenceEngine generation and lifecycle."""

    def __init__(self, response_chunks: list[str] | None = None, initial_id: str | None = None):
        self.chunks = response_chunks or ["Hello", " world", "!"]
        self._id = initial_id
        self.was_stopped = False

    @property
    def loaded_model_id(self) -> str | None:
        return self._id

    def load(self, spec: LoadSpec):
        self._id = spec.model_id

    def unload(self):
        self._id = None

    def generate(self, messages, params, on_token, should_stop):
        out = []
        for chunk in self.chunks:
            time.sleep(0.01)
            if should_stop():
                self.was_stopped = True
                return GenerationResult("".join(out), len(out), 0.05, stopped=True)
            on_token(chunk)
            out.append(chunk)
        return GenerationResult("".join(out), len(out), 0.05, stopped=False)


@pytest.fixture
def test_entry():
    return create_test_model_entry(model_id="test-model")


@pytest.fixture
def initialized_backend(paths, test_entry, tmp_path):
    """Fixture providing a ready Backend with one installed model and fake engine."""
    backend = Backend(paths)

    # Setup fake installed model file
    model_file = paths.models / "test-model" / "model.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"dummy model data")

    installed_entry = InstalledModel(
        entry=test_entry,
        directory=model_file.parent,
        installed_at="2026-11-18T10:00:00Z",
    )

    fake_engine = FakeTestEngine(initial_id=test_entry.id)

    # Pre-populate backend initialized state
    backend.registry = ModelRegistry([test_entry])
    backend.store = ModelStore(paths, backend.registry)
    backend.hardware = HardwareInfo(
        arch="x86_64",
        chip="MockChip",
        total_ram_bytes=16 * 1024**3,
        cpu_cores=8,
        apple_silicon=False,
        metal=False,
        os_name="macOS",
        os_major=15,
    )
    backend._installed = {test_entry.id: installed_entry}
    backend._runtime_ok = True
    backend._runtime_version = "mock-1.0"
    backend._ready = True

    # Inject factory so reloading/unloading creates fresh engines properly
    backend.runtime._engine = fake_engine
    backend.runtime._backend = "fake"
    backend.runtime._factory = lambda b: FakeTestEngine(initial_id=test_entry.id)
    backend.runtime._loaded_spec = LoadSpec(model_id=test_entry.id, model_path=model_file)

    yield backend, fake_engine
    backend.shutdown()


class TestSendMessageLifecycle:
    """Test Backend.sendMessage -> streamed response -> persisted reply."""

    def test_send_message_creates_conversation_and_persists_messages(self, initialized_backend):
        backend, _fake_engine = initialized_backend

        # Send a message
        ok = backend.sendMessage("Hello there", [])
        assert ok is True

        # Wait for generation worker thread to finish
        if backend._gen_thread:
            backend._gen_thread.join(timeout=3.0)

        QGuiApplication.processEvents()

        # Verify conversation was created
        assert backend.currentConversationId != ""
        conv = backend.db.get_conversation(backend.currentConversationId)
        assert conv is not None

        # Verify database messages (user and assistant)
        msgs = backend.db.messages(conv.id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello there"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Hello world!"
        assert msgs[1].model_id == "test-model"
        assert msgs[1].metadata.get("stopped") is False
        assert msgs[1].metadata.get("tokens") == 3

        # Verify UI message model
        assert backend.messages.count == 2
        assert backend.messages.items()[0]["role"] == "user"
        assert backend.messages.items()[1]["role"] == "assistant"
        assert backend.messages.items()[1]["content"] == "Hello world!"
        assert backend.messages.items()[1]["streaming"] is False

    def test_send_message_not_ready_or_empty_returns_false(self, initialized_backend):
        backend, _ = initialized_backend

        # Empty message
        assert backend.sendMessage("   ", []) is False

        # Not ready
        backend._ready = False
        assert backend.sendMessage("Hello", []) is False

    def test_send_message_runtime_not_ok_emits_toast(self, initialized_backend):
        backend, _ = initialized_backend
        backend._runtime_ok = False

        toast_mock = MagicMock()
        backend.toast.connect(toast_mock)

        assert backend.sendMessage("Hello", []) is False
        toast_mock.assert_called_once()
        assert "runtime failed" in toast_mock.call_args[0][0]


class TestStopGeneration:
    """Test stopGeneration behaviour."""

    def test_stop_generation_halts_stream(self, initialized_backend):
        backend, _ = initialized_backend

        # Setup engine with many chunks
        slow_engine = FakeTestEngine(response_chunks=["chunk"] * 50, initial_id="test-model")
        backend.runtime._engine = slow_engine

        ok = backend.sendMessage("Start long generation", [])
        assert ok is True

        # Wait slightly then call stopGeneration
        time.sleep(0.02)
        backend.stopGeneration()

        if backend._gen_thread:
            backend._gen_thread.join(timeout=3.0)

        QGuiApplication.processEvents()

        conv = backend.db.get_conversation(backend.currentConversationId)
        msgs = backend.db.messages(conv.id)
        assert len(msgs) == 2
        assert msgs[1].metadata.get("stopped") is True

    def test_stop_generation_when_idle_is_safe(self, initialized_backend):
        backend, _ = initialized_backend
        assert not backend.is_generating
        # Must not raise
        backend.stopGeneration()
        assert not backend.is_generating


class TestRegenerate:
    """Test regenerate behaviour."""

    def test_regenerate_replaces_last_assistant_reply(self, initialized_backend):
        backend, fake_engine = initialized_backend

        backend.sendMessage("Tell me a fact", [])
        if backend._gen_thread:
            backend._gen_thread.join(timeout=3.0)

        QGuiApplication.processEvents()
        assert backend.messages.count == 2

        # Change engine chunks for regenerated response
        fake_engine.chunks = ["A", " new", " fact!"]

        backend.regenerate()
        if backend._gen_thread:
            backend._gen_thread.join(timeout=3.0)

        QGuiApplication.processEvents()

        # Message count should still be 2 (old assistant replaced)
        msgs = backend.db.messages(backend.currentConversationId)
        assert len(msgs) == 2
        assert msgs[1].content == "A new fact!"
        assert backend.messages.items()[1]["content"] == "A new fact!"

    def test_regenerate_when_generating_or_empty_is_noop(self, initialized_backend):
        backend, _ = initialized_backend
        # In a new chat with no messages
        backend.newChat()
        backend.regenerate()
        assert backend.messages.count == 0


class TestDeleteConversation:
    """Test deleteConversation cleans up database, messages, and attachments."""

    def test_delete_conversation_cleans_attachments(self, initialized_backend, tmp_path):
        backend, _ = initialized_backend

        # Prepare fake image attachment in backend attachments folder
        att_dir = backend.paths.attachments
        att_file = att_dir / "test_image.png"
        att_file.write_bytes(b"image png data")

        # Create conversation with attachment
        conv = backend.db.create_conversation("Chat with Image")
        backend.db.add_message(conv.id, "user", "Look", metadata={"images": [str(att_file)]})

        backend.openConversation(conv.id)
        assert att_file.exists()

        # Delete conversation
        backend.deleteConversation(conv.id)

        # Verify DB and attachment deleted
        assert backend.db.get_conversation(conv.id) is None
        assert not att_file.exists()
        assert backend.currentConversationId == ""

    def test_delete_conversation_preserves_external_files(self, initialized_backend, tmp_path):
        backend, _ = initialized_backend

        external_file = tmp_path / "outside.png"
        external_file.write_bytes(b"external")

        conv = backend.db.create_conversation("Chat with External File")
        backend.db.add_message(conv.id, "user", "Look", metadata={"images": [str(external_file)]})

        backend.deleteConversation(conv.id)

        assert backend.db.get_conversation(conv.id) is None
        # File outside attachments folder must NOT be deleted
        assert external_file.exists()


class TestDeleteBlockersAndModel:
    """Test deleteBlocker and deleteModel behaviors."""

    def test_delete_blocker_when_model_is_loaded(self, initialized_backend, test_entry):
        backend, _ = initialized_backend

        blocker = backend.deleteBlocker(test_entry.id)
        assert "currently loaded" in blocker

        # deleteModel should fail and show toast
        toast_mock = MagicMock()
        backend.toast.connect(toast_mock)
        success = backend.deleteModel(test_entry.id)
        assert success is False
        toast_mock.assert_called_once()
        assert "currently loaded" in toast_mock.call_args[0][0]

    def test_delete_blocker_when_model_is_generating(self, initialized_backend, test_entry):
        backend, _ = initialized_backend

        slow_engine = FakeTestEngine(response_chunks=["chunk"] * 100, initial_id="test-model")
        backend.runtime._engine = slow_engine
        backend.sendMessage("Test", [])

        try:
            assert backend.is_generating
            blocker = backend.deleteBlocker(test_entry.id)
            assert "generating a response" in blocker
        finally:
            backend.stopGeneration()
            if backend._gen_thread:
                backend._gen_thread.join(timeout=3.0)

    def test_delete_model_success_when_unloaded(self, initialized_backend, test_entry):
        backend, _ = initialized_backend

        # Unload the model first
        backend.runtime.unload()
        assert backend.runtime.loaded_model_id is None

        blocker = backend.deleteBlocker(test_entry.id)
        assert blocker == ""

        success = backend.deleteModel(test_entry.id)
        assert success is True
        assert test_entry.id not in backend._installed


class TestManualModeDeletedModelError:
    """Test manual mode error reporting when chosen model is not downloaded."""

    def test_manual_mode_missing_model_error(self, initialized_backend):
        backend, _ = initialized_backend

        # Create conversation with a missing model in manual mode
        conv = backend.db.create_conversation("Manual Missing", "nonexistent-model", "manual")
        backend.openConversation(conv.id)

        assert backend.selectedModelMissing is True

        # Attempt to trigger generation
        backend._start_generation(conv.id)

        # Should append an error assistant message
        msgs = backend.db.messages(conv.id)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].metadata.get("error") is True
        assert "not downloaded anymore" in msgs[0].content


# =====================================================================
# Phase 6 — Settings Tests
# =====================================================================


class TestSettings:
    """Test setSetting type coercion, persistence, and model unloading triggers."""

    def test_settings_type_coercion_and_persistence(self, paths):
        backend = Backend(paths)

        # Integer coercion
        backend.setSetting("max_tokens", "2048")
        assert backend.settings.max_tokens == 2048
        assert backend.maxTokens == 2048

        # Float coercion
        backend.setSetting("temperature", "0.2")
        assert backend.settings.temperature == pytest.approx(0.2)
        assert backend.temperature == pytest.approx(0.2)

        # Boolean coercion
        backend.setSetting("enable_thinking", True)
        assert backend.settings.enable_thinking is True
        assert backend.enableThinking is True

        # String setting
        backend.setSetting("accent", "green")
        assert backend.settings.accent == "green"
        assert backend.accent == "green"

        # Verify disk persistence
        reloaded = Settings.load(paths.settings_file)
        assert reloaded.max_tokens == 2048
        assert reloaded.temperature == pytest.approx(0.2)
        assert reloaded.enable_thinking is True
        assert reloaded.accent == "green"

        backend.shutdown()

    def test_unknown_or_invalid_settings_ignored(self, paths):
        backend = Backend(paths)
        initial_tokens = backend.settings.max_tokens

        # Unknown setting
        backend.setSetting("nonexistent_setting", "value")

        # Invalid type conversion (e.g. integer given invalid string)
        backend.setSetting("max_tokens", "not-a-number")
        assert backend.settings.max_tokens == initial_tokens

        backend.shutdown()

    def test_context_length_and_use_gpu_trigger_model_unload(self, initialized_backend, test_entry):
        backend, _ = initialized_backend
        assert backend.runtime.loaded_model_id == "test-model"

        # Changing context_length must trigger unloadModel()
        backend.setSetting("context_length", 8192)
        # Give worker thread a moment to unload
        for _ in range(20):
            if backend.runtime.loaded_model_id is None:
                break
            time.sleep(0.05)
        assert backend.runtime.loaded_model_id is None

        # Re-set loaded model using the fake engine backend
        model_file = backend.paths.models / test_entry.id / "model.gguf"
        backend.runtime.ensure_loaded(
            LoadSpec(model_id=test_entry.id, model_path=model_file), "fake"
        )
        assert backend.runtime.loaded_model_id == "test-model"

        # Changing use_gpu must also trigger unloadModel()
        backend.setSetting("use_gpu", False)
        for _ in range(20):
            if backend.runtime.loaded_model_id is None:
                break
            time.sleep(0.05)
        assert backend.runtime.loaded_model_id is None

    def test_unrelated_setting_does_not_unload_model(self, initialized_backend):
        backend, _ = initialized_backend
        assert backend.runtime.loaded_model_id == "test-model"

        # Changing temperature should NOT unload
        backend.setSetting("temperature", 0.9)
        assert backend.runtime.loaded_model_id == "test-model"


# =====================================================================
# Additional Backend Unit Tests (DictListModel, Navigation, Rename)
# =====================================================================


class TestDictListModel:
    """Test DictListModel data access, append, update, remove, and find."""

    def test_dict_list_model_crud(self):
        model = DictListModel(["id", "name", "value"])
        assert model.count == 0

        # Append
        model.append({"id": "1", "name": "Item 1", "value": 100})
        model.append({"id": "2", "name": "Item 2", "value": 200})
        assert model.count == 2
        assert model.get(0) == {"id": "1", "name": "Item 1", "value": 100}

        # Find
        assert model.find("id", "2") == 1
        assert model.find("id", "3") == -1

        # Update
        model.update(1, {"value": 250})
        assert model.get(1)["value"] == 250

        # Remove
        model.remove(0)
        assert model.count == 1
        assert model.get(0)["id"] == "2"

        # Reset
        model.reset([{"id": "a", "name": "A", "value": 1}])
        assert model.count == 1
        assert model.get(0)["id"] == "a"


class TestConversationManagement:
    """Test rename, newChat, openConversation, and search."""

    def test_rename_and_search(self, initialized_backend):
        backend, _ = initialized_backend

        conv = backend.db.create_conversation("Old Title")
        backend.openConversation(conv.id)
        assert backend.currentTitle == "Old Title"

        backend.renameConversation(conv.id, "New Brand Title")
        assert backend.currentTitle == "New Brand Title"

        # Search
        backend.setSearch("Brand")
        assert backend.conversations.count == 1

        backend.setSearch("NonExistent")
        assert backend.conversations.count == 0

        # New chat resets
        backend.newChat()
        assert backend.currentConversationId == ""
        assert backend.currentTitle == "New chat"

def test_queued_download_can_be_cancelled(initialized_backend):
    backend, _ = initialized_backend

    class FakeDownloader:
        cancelled = False
        files_to_download = []

        def download(self):
            pytest.fail("Download should not start after cancellation")

    downloader = FakeDownloader()
    errors = []

    backend._download_lock.acquire()

    def worker():
        try:
            backend._download_with_lock("test-model", downloader)
        except DownloadCancelled:
            errors.append("cancelled")

    thread = threading.Thread(target=worker)
    try:
        thread.start()
        time.sleep(0.2)
        downloader.cancelled = True
        thread.join(timeout=2)
    finally:
        backend._download_lock.release()

    assert not thread.is_alive()
    assert errors == ["cancelled"]