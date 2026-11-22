"""Tests for single-instance protection of a data folder (Issue #11)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

from app.config import AppPaths
from app.instance_lock import InstanceLock, activation_name

ROOT = Path(__file__).resolve().parent.parent


def test_second_lock_on_same_folder_is_refused(paths: AppPaths):
    first, second = InstanceLock(paths.data), InstanceLock(paths.data)
    assert first.acquire()
    assert not second.acquire()
    assert not second.held
    first.release()


def test_lock_can_be_retaken_after_release(paths: AppPaths):
    first = InstanceLock(paths.data)
    assert first.acquire()
    first.release()
    assert InstanceLock(paths.data).acquire()


def test_acquire_is_idempotent(paths: AppPaths):
    lock = InstanceLock(paths.data)
    assert lock.acquire()
    assert lock.acquire()
    lock.release()


def test_different_folders_do_not_conflict(tmp_path: Path):
    a, b = InstanceLock(tmp_path / "a"), InstanceLock(tmp_path / "b")
    assert a.acquire()
    assert b.acquire()
    a.release()
    b.release()


def test_stale_lock_file_does_not_block(paths: AppPaths):
    # A leftover file (e.g. from a crash) with a dead PID in it, but no live flock.
    (paths.data / ".lock").write_text("999999\n")
    assert InstanceLock(paths.data).acquire()


def test_lock_from_killed_process_is_released(paths: AppPaths):
    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from pathlib import Path
        from app.instance_lock import InstanceLock
        lock = InstanceLock(Path({str(paths.data)!r}))
        assert lock.acquire()
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        assert not InstanceLock(paths.data).acquire()  # held by the live process
        proc.send_signal(signal.SIGKILL)  # crash: no cleanup runs
        proc.wait(timeout=10)
        assert InstanceLock(paths.data).acquire()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_activation_name_is_per_folder(tmp_path: Path):
    assert activation_name(tmp_path / "a") == activation_name(tmp_path / "a")
    assert activation_name(tmp_path / "a") != activation_name(tmp_path / "b")
    assert os.sep not in activation_name(tmp_path / "a")
