"""Single-instance protection for a SmolChat data folder.

Two processes sharing one data folder would both write ``chats.sqlite3``,
``settings.json`` and the partial-download files. :class:`InstanceLock` takes an
exclusive ``flock`` on ``<data>/.lock`` so only one process per data folder runs.

The kernel drops an ``flock`` when its process exits, however it exits, so a lock
left behind by a crash never blocks the next launch. Different ``SMOLCHAT_HOME``
folders have different lock files and can run side by side.

This module is Qt-free (see CONTRIBUTING.md); window activation lives in ``main.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path

LOCK_FILENAME = ".lock"


class InstanceLock:
    """Exclusive, non-blocking lock on a data folder."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / LOCK_FILENAME
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Try to take the lock. Returns False if another process holds it."""
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        # The PID is informational only; the flock is what guarantees exclusivity.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def activation_name(data_dir: Path) -> str:
    """Per-data-folder name for the local socket used to raise the running window."""
    digest = hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()[:16]
    return f"smolchat-{digest}"
