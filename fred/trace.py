"""Weave tracing, made optional.

Every Cosmos call, every anchor scan and every agent answer is a traced op when Weave is
available, so a judge can click into the run and see the clip, the prompt and the model's
reasoning. When Weave or its credentials are absent the decorator degrades to a no-op and
nothing else in the codebase has to care.
"""
from __future__ import annotations

import functools
import os

from . import config

_state = {"inited": False, "enabled": False, "project": None, "error": ""}


def init(project: str = None) -> dict:
    """Initialise Weave once. Safe to call repeatedly."""
    if _state["inited"]:
        return dict(_state)
    _state["inited"] = True
    if os.environ.get("FRED_DISABLE_WEAVE"):
        _state["error"] = "disabled by FRED_DISABLE_WEAVE"
        return dict(_state)
    try:
        import weave  # noqa: F401

        team = config.get("WANDB_TEAM")
        name = project or config.get("WANDB_PROJECT") or "fred-lasso"
        full = f"{team}/{name}" if team else name
        api_key = config.get("WANDB_API_KEY")
        if api_key:
            os.environ.setdefault("WANDB_API_KEY", api_key)
        weave.init(full)
        _state.update({"enabled": True, "project": full})
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{type(exc).__name__}: {exc}"
    return dict(_state)


def op(func=None, **kwargs):
    """@trace.op - a weave.op when Weave is live, otherwise the function unchanged."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            return wrapper._impl(*args, **kw)

        def _resolve():
            if not _state["inited"]:
                init()
            if not _state["enabled"]:
                return fn
            try:
                import weave

                return weave.op(fn, **kwargs) if kwargs else weave.op(fn)
            except Exception:  # noqa: BLE001
                return fn

        wrapper._impl = fn

        @functools.wraps(fn)
        def lazy(*args, **kw):
            if getattr(lazy, "_bound", None) is None:
                lazy._bound = _resolve()
            return lazy._bound(*args, **kw)

        lazy._bound = None
        return lazy

    return decorate(func) if callable(func) else decorate


def status() -> dict:
    return dict(_state)
