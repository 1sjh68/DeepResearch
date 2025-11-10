"""
Utilities for persisting intermediate iteration outputs to the session directory.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from config import Config


def _ensure_iteration_dir(config: Config) -> str | None:
    """
    Ensure the iteration archive directory exists for the current session.
    """
    base_dir = getattr(config, "session_dir", "") or ""
    if not base_dir:
        logging.debug("Iteration archive skipped: config.session_dir is not set yet.")
        return None

    archive_dir = os.path.join(base_dir, "iterations")
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir


def archive_iteration_snapshot(
    config: Config,
    iteration_index: int,
    stage_name: str,
    content: str,
) -> str | None:
    """
    Persist the given content as an iteration snapshot for rollback/reference.

    Args:
        config: Active Config instance with session metadata.
        iteration_index: The zero-based iteration counter (0 for initial draft).
        stage_name: Logical stage label, e.g. "initial_draft", "refine".
        content: Markdown text to persist.

    Returns:
        Absolute path to the saved snapshot, or None if skipped.
    """
    if not content or not content.strip():
        logging.debug(
            "Iteration archive skipped: empty content for iteration %s (%s).",
            iteration_index,
            stage_name,
        )
        return None

    archive_dir = _ensure_iteration_dir(config)
    if not archive_dir:
        return None

    safe_stage = stage_name.strip().lower().replace(" ", "_").replace("-", "_")
    filename = f"iter_{iteration_index:02d}_{safe_stage}.md"
    file_path = os.path.join(archive_dir, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            header = f"<!-- iteration: {iteration_index}, stage: {safe_stage}, saved_at: {datetime.now().isoformat()} -->\n"
            file.write(header)
            file.write(content)
        logging.info(
            "Iteration snapshot saved: %s (iteration=%s, stage=%s)",
            file_path,
            iteration_index,
            safe_stage,
        )
        return file_path
    except OSError as exc:
        logging.error(
            "Failed to write iteration snapshot for iteration %s (%s): %s",
            iteration_index,
            stage_name,
            exc,
        )
        return None
