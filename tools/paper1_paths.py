"""Portable path helpers for Paper 1 reproduction scripts."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path("dataset/ag_data/data/world_model/quentinll")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def paper1_data_root() -> Path:
    """Return the root that contains the released ``lewm-*`` task folders.

    Prefer ``PAPER1_DATA_ROOT``.  ``STABLEWM_HOME`` is also accepted for
    compatibility; if it points at a task folder such as ``lewm-pusht``, this
    function returns its parent.
    """

    raw = os.environ.get("PAPER1_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()

    raw = os.environ.get("STABLEWM_HOME")
    if raw:
        path = Path(raw).expanduser()
        if path.name.startswith("lewm-"):
            return path.parent
        return path

    return repo_root() / DEFAULT_DATA_ROOT


def default_model_roots() -> list[Path]:
    root = paper1_data_root()
    return [root] if str(root) else []


def task_dir(task_root: str) -> Path:
    return paper1_data_root() / task_root


def portable_path(path: Path, root: Path | None = None) -> str:
    """Store checkpoint paths relative to the data root when possible."""

    root = paper1_data_root() if root is None else root
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
