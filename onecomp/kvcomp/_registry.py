"""Lightweight registries for KV compression methods.

KVCache-Factory selects methods via a large ``if/elif`` ladder and global
monkeypatching (``replace_llama``).  OneComp instead follows the registry
pattern so that adding a method is a single decorator and methods are
discoverable / configurable by name (mirroring how AutoBit dispatches to many
child quantizers).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Callable, Type

_KV_CODECS: dict[str, Type] = {}
_KV_SELECTORS: dict[str, Type] = {}
_KV_POLICIES: dict[str, Type] = {}


def _make_register(table: dict[str, Type], kind: str) -> Callable:
    def register(name: str) -> Callable:
        def deco(cls):
            if name in table and table[name] is not cls:
                raise ValueError(f"{kind} '{name}' is already registered to {table[name]!r}")
            table[name] = cls
            # Record the registry key so instances serialise under it (not the
            # class name) -- see ``*.__post_init__`` and ``get_kv_config``.
            cls._registry_name = name
            return cls

        return deco

    return register


register_kv_codec = _make_register(_KV_CODECS, "KV codec")
register_kv_selector = _make_register(_KV_SELECTORS, "KV selector")
register_kv_policy = _make_register(_KV_POLICIES, "KV policy")


def _build(table: dict[str, Type], kind: str, name: str, **kwargs):
    if name not in table:
        available = ", ".join(sorted(table)) or "<none>"
        raise KeyError(f"Unknown {kind} '{name}'. Registered: {available}")
    return table[name](**kwargs)


def build_codec(name: str, **kwargs):
    """Instantiate a registered codec by name."""
    return _build(_KV_CODECS, "KV codec", name, **kwargs)


def build_selector(name: str, **kwargs):
    """Instantiate a registered selector by name."""
    return _build(_KV_SELECTORS, "KV selector", name, **kwargs)


def build_policy(name: str, **kwargs):
    """Instantiate a registered policy by name."""
    return _build(_KV_POLICIES, "KV policy", name, **kwargs)


def list_codecs() -> list[str]:
    """Return the names of all registered codecs."""
    return sorted(_KV_CODECS)


def list_selectors() -> list[str]:
    """Return the names of all registered selectors."""
    return sorted(_KV_SELECTORS)


def list_policies() -> list[str]:
    """Return the names of all registered policies."""
    return sorted(_KV_POLICIES)
