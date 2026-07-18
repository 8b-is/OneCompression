"""CPU-only fail-closed tests for full HQ registration in vLLM children."""

from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace
import sys

import pytest

from vllm_plugins.kvcomp import vllm_plugin


def _fake_registry(monkeypatch):
    calls = []
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = SimpleNamespace(TURBOQUANT="TURBOQUANT")
    registry.register_backend = lambda enum, path: calls.append((enum, path))
    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    return calls


def test_no_explicit_full_env_is_a_strict_noop(monkeypatch):
    monkeypatch.delenv(vllm_plugin._HAYATE_FULL_NATIVE_CONFIG_ENV, raising=False)
    monkeypatch.setattr(vllm_plugin, "_HAYATE_FULL_NATIVE_INSTALLED", False)
    assert vllm_plugin._register_hayate_full_native_backend(SimpleNamespace()) is False


def test_valid_explicit_full_env_registers_exact_backend(monkeypatch):
    calls = _fake_registry(monkeypatch)
    monkeypatch.delenv(vllm_plugin._HAYATE_NATIVE_CONFIG_ENV, raising=False)
    monkeypatch.setenv(
        vllm_plugin._HAYATE_FULL_NATIVE_CONFIG_ENV,
        json.dumps({"avg_bits": 3, "recipe": "hq"}),
    )
    monkeypatch.setattr(vllm_plugin, "_HAYATE_FULL_NATIVE_INSTALLED", False)
    logger = SimpleNamespace(info=lambda *args: None)
    assert vllm_plugin._register_hayate_full_native_backend(logger) is True
    assert calls == [
        (
            "TURBOQUANT",
            "vllm_plugins.kvcomp.hayate_full_native.FullHayateQuantNativeBackend",
        )
    ]


def test_malformed_or_ambiguous_explicit_env_fails_closed(monkeypatch):
    monkeypatch.setattr(vllm_plugin, "_HAYATE_FULL_NATIVE_INSTALLED", False)
    monkeypatch.setenv(vllm_plugin._HAYATE_FULL_NATIVE_CONFIG_ENV, "not-json")
    monkeypatch.delenv(vllm_plugin._HAYATE_NATIVE_CONFIG_ENV, raising=False)
    with pytest.raises(RuntimeError, match="invalid explicit"):
        vllm_plugin._register_hayate_full_native_backend(SimpleNamespace())
    monkeypatch.setenv(vllm_plugin._HAYATE_FULL_NATIVE_CONFIG_ENV, '{"avg_bits": 3}')
    monkeypatch.setenv(vllm_plugin._HAYATE_NATIVE_CONFIG_ENV, '{"bits": 3}')
    with pytest.raises(RuntimeError, match="ambiguous"):
        vllm_plugin._register_hayate_full_native_backend(SimpleNamespace())
