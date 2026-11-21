"""Qt-independent conversation export helpers.

Converts conversation data into Markdown or JSON without importing Qt,
any database connection, or other UI classes.  Call-sites are responsible
for fetching the conversation and messages from the database and passing
them in as plain Python dicts / dataclasses.

Expected call signature (pseudo-code):
    from app.export import conversation_to_markdown, conversation_to_json

    conv = db.get_conversation(conv_id)
    msgs = db.messages(conv_id)
    md_text  = conversation_to_markdown(conv, msgs)
    json_text = conversation_to_json(conv, msgs)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)")


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>…</think>`` blocks (or unclosed opening tags).

    Mirrors the logic in ``app.formatting.strip_reasoning`` but lives here
    so the export module stays Qt-free and independently testable.

    Rules:
    - A complete ``<think>…</think>`` pair is removed in full.
    - An opening ``<think>`` with **no** matching ``</think>`` is treated as
      a reasoning block that runs to the end of the string; the entire tail
      is removed so we never accidentally leak partial thought text.
    - Text before the opening tag is preserved.
    - Multiple blocks in a single message are all removed.
    """
    return _THINK_RE.sub("", text).strip()


def _fmt_ts(unix_ts: float) -> str:
    """Return an ISO-8601 UTC timestamp string from a Unix float."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_filename(title: str) -> str:
    """Turn a conversation title into a reasonable default file base name."""
    # Replace runs of non-alphanumeric chars with a single hyphen.
    name = re.sub(r"[^\w\s-]", "", title).strip()
    name = re.sub(r"[\s_-]+", "-", name)
    return name[:80] or "conversation"


# ---------------------------------------------------------------------------
# Public API — Markdown export
# ---------------------------------------------------------------------------


def conversation_to_markdown(conversation: Any, messages: list[Any]) -> str:
    """Serialise *conversation* and *messages* to a Markdown string.

    Parameters
    ----------
    conversation:
        A ``Conversation`` dataclass (or any object with ``.title``,
        ``.created_at``, ``.id``) from ``app.database``.
    messages:
        Ordered list of ``Message`` dataclasses (or objects with ``.role``,
        ``.content``, ``.created_at``, ``.model_id``, ``.metadata``).

    Returns
    -------
    str
        UTF-8 Markdown text ready to write to a ``.md`` file.
    """
    lines: list[str] = []

    title = conversation.title or "Untitled chat"
    created_ts = _fmt_ts(conversation.created_at)

    lines.append(f"# {title}")
    lines.append("")
    lines.append("**Exported from SmolChat**  ")
    lines.append(f"*Created: {created_ts}*  ")
    lines.append(f"*Conversation ID: {conversation.id}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = getattr(msg, "role", "")
        if role == "system":
            # System messages are not part of the visible conversation.
            continue

        content: str = getattr(msg, "content", "") or ""
        model_id: str = getattr(msg, "model_id", "") or ""
        meta: dict = getattr(msg, "metadata", {}) or {}
        ts = _fmt_ts(getattr(msg, "created_at", 0.0))

        if role == "user":
            lines.append("**You**")
            lines.append(f"*{ts}*")
            lines.append("")

            # Image attachments
            images: list[str] = meta.get("images", [])
            if images:
                for img in images:
                    # Only include the filename; don't expose internal paths.
                    fname = os.path.basename(img)
                    lines.append(f"📎 *Attachment: {fname}*")
                lines.append("")

            lines.append(content)

        elif role == "assistant":
            model_label = model_id if model_id else "SmolChat"
            lines.append(f"**SmolChat ({model_label})**")
            lines.append(f"*{ts}*")
            lines.append("")

            # Strip <think> reasoning blocks before exporting.
            exported_content = _strip_reasoning(content)
            lines.append(exported_content)

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — JSON export
# ---------------------------------------------------------------------------


def conversation_to_json(conversation: Any, messages: list[Any]) -> str:
    """Serialise *conversation* and *messages* to a pretty-printed JSON string.

    Parameters
    ----------
    conversation:
        A ``Conversation`` dataclass from ``app.database``.
    messages:
        Ordered list of ``Message`` dataclasses from ``app.database``.

    Returns
    -------
    str
        UTF-8 JSON text with 2-space indentation, ready to write to a
        ``.json`` file.

    Notes
    -----
    - All fields that exist in the dataclasses are preserved.
    - No fields are invented.
    - Timestamps are stored both as the original Unix float (``created_at``)
      *and* as a human-readable ISO-8601 string (``created_at_iso``) for
      convenience.
    - ``metadata`` is kept as-is (already a plain dict).
    - ``<think>`` blocks are **not** stripped from the JSON export; the JSON
      is a faithful archive of the raw stored content.
    """
    conv_dict: dict[str, Any] = {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "created_at_iso": _fmt_ts(conversation.created_at),
        "updated_at": conversation.updated_at,
        "updated_at_iso": _fmt_ts(conversation.updated_at),
        "model_id": conversation.model_id,
        "model_mode": conversation.model_mode,
    }

    msg_list: list[dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "role", "")
        meta: dict = getattr(msg, "metadata", {}) or {}

        entry: dict[str, Any] = {
            "id": getattr(msg, "id", None),
            "conversation_id": getattr(msg, "conversation_id", conversation.id),
            "role": role,
            "content": getattr(msg, "content", ""),
            "created_at": getattr(msg, "created_at", 0.0),
            "created_at_iso": _fmt_ts(getattr(msg, "created_at", 0.0)),
            "model_id": getattr(msg, "model_id", ""),
            "metadata": meta,
        }
        msg_list.append(entry)

    export: dict[str, Any] = {
        "smolchat_export_version": 1,
        "conversation": conv_dict,
        "messages": msg_list,
    }

    return json.dumps(export, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def default_filename(conversation: Any, fmt: str) -> str:
    """Return a sensible default save filename for the conversation.

    Parameters
    ----------
    conversation:
        Any object with a ``.title`` string attribute.
    fmt:
        ``"md"`` or ``"json"``.

    Returns
    -------
    str  — e.g. ``"My-Project-Chat.md"``
    """
    base = _safe_filename(getattr(conversation, "title", "") or "conversation")
    return f"{base}.{fmt}"
