"""Unit tests for RoleKV: role spans, precision policy, and the graded cache.

CPU-only, model-free.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp.kvcomp import build_policy, list_policies
from onecomp.kvcomp.cache.role_cache import RoleKVCache
from onecomp.kvcomp.recipe.rolekv import (
    RolePrecisionPolicy,
    RoleSpan,
    annotate_spans_from_chat,
    role_token_bits,
    validate_spans,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class _FakeTokenizer:
    """Chat-template stub: each message contributes ``10 + len(content) // 4`` tokens."""

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=False
    ):
        total = sum(10 + len(m.get("content", "")) // 4 for m in messages)
        return list(range(total))


# --- spans and policy -------------------------------------------------------
def test_role_span_validation():
    with pytest.raises(ValueError):
        RoleSpan(5, 3, "system")
    with pytest.raises(ValueError):
        RoleSpan(0, 3, "narrator")


def test_policy_registered_and_defaults():
    assert "rolekv" in list_policies()
    policy = build_policy("rolekv")
    assert isinstance(policy, RolePrecisionPolicy)
    assert policy.bits_for("system") == 8
    assert policy.bits_for("user") == 8
    assert policy.bits_for("assistant") == 4
    assert policy.bits_for("reasoning") == 4
    assert policy.bits_for("tool") == 2
    assert policy.bits_for("unknown-role") == policy.default_bits
    assert policy.max_bits() == 8


def test_policy_validation():
    with pytest.raises(ValueError):
        RolePrecisionPolicy(bits_per_role={"system": 5})
    with pytest.raises(ValueError):
        RolePrecisionPolicy(default_bits=7)


def test_policy_serialization_roundtrip():
    policy = RolePrecisionPolicy(
        bits_per_role={"system": 8, "tool": 2}, default_bits=4, sink_tokens=8
    )
    cfg = policy.get_kv_config()
    assert cfg["name"] == "rolekv"
    rebuilt = build_policy(cfg.pop("name"), **cfg)
    assert rebuilt.bits_per_role == {"system": 8, "tool": 2}
    assert rebuilt.default_bits == 4
    assert rebuilt.sink_tokens == 8


def test_validate_spans_tiling():
    spans = [RoleSpan(16, 32, "tool"), RoleSpan(0, 16, "system")]
    ordered = validate_spans(spans, 32)
    assert [s.start for s in ordered] == [0, 16]
    with pytest.raises(ValueError):
        validate_spans([RoleSpan(0, 16, "system")], 32)
    with pytest.raises(ValueError):
        validate_spans([RoleSpan(0, 16, "system"), RoleSpan(20, 32, "tool")], 32)
    with pytest.raises(ValueError):
        validate_spans([], 32)


def test_annotate_spans_cover_all_tokens():
    messages = [
        {"role": "system", "content": "You are a helpful agent." * 3},
        {"role": "user", "content": "What is the weather in Kawasaki?"},
        {"role": "tool", "content": "temperature=31C humidity=70%" * 5},
        {"role": "assistant", "content": "It is hot."},
    ]
    tokenizer = _FakeTokenizer()
    ids = tokenizer.apply_chat_template(messages)
    spans = annotate_spans_from_chat(messages, tokenizer, ids)

    assert spans[0].start == 0
    assert spans[-1].end == len(ids)
    for prev, cur in zip(spans, spans[1:]):
        assert prev.end == cur.start
    assert [s.role for s in spans] == ["system", "user", "tool", "assistant"]


def test_annotate_spans_unknown_role_falls_back_to_user():
    messages = [{"role": "function", "content": "x" * 40}, {"role": "assistant", "content": "ok"}]
    tokenizer = _FakeTokenizer()
    ids = tokenizer.apply_chat_template(messages)
    spans = annotate_spans_from_chat(messages, tokenizer, ids)
    assert spans[0].role == "user"


# --- bit schedule ------------------------------------------------------------
def test_role_token_bits_schedule():
    policy = RolePrecisionPolicy(
        bits_per_role={"system": 8, "tool": 2, "assistant": 4}, sink_tokens=0
    )
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 32, "tool")]
    bits = role_token_bits(spans, 40, policy, device="cpu")
    assert bits.shape == (40,)
    assert (bits[:16] == 8).all()
    assert (bits[16:32] == 2).all()
    # Positions past the annotated prefix are generated tokens -> assistant.
    assert (bits[32:] == 4).all()


def test_role_token_bits_protects_sink_tokens():
    """The leading sink tokens stay in full precision regardless of role."""
    policy = RolePrecisionPolicy(bits_per_role={"system": 2, "tool": 2}, sink_tokens=4)
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 32, "tool")]
    bits = role_token_bits(spans, 32, policy, device="cpu")
    assert (bits[:4] == 16).all()
    assert (bits[4:16] == 2).all()


def test_role_token_bits_clips_to_current_length():
    policy = RolePrecisionPolicy(sink_tokens=0)
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 32, "tool")]
    bits = role_token_bits(spans, 20, policy, device="cpu")
    assert bits.shape == (20,)
    assert (bits[:16] == 8).all()
    assert (bits[16:] == 2).all()


# --- live graded cache --------------------------------------------------------
def test_role_cache_grades_spans_differently():
    b, h, s, d = 1, 2, 48, 64
    policy = RolePrecisionPolicy(bits_per_role={"system": 8, "tool": 2}, sink_tokens=0)
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 48, "tool")]
    cache = RoleKVCache(policy, spans, requant_stride=1)

    key = torch.randn(b, h, s, d)
    value = torch.randn(b, h, s, d)
    out_k, _ = cache.update(key.clone(), value.clone(), 0)

    system_err = (out_k[:, :, :16] - key[:, :, :16]).abs().mean().item()
    tool_err = (out_k[:, :, 16:] - key[:, :, 16:]).abs().mean().item()
    assert system_err < tool_err
    assert cache.compression_ratio() > 1.0


def test_role_cache_does_not_mutate_caller_spans():
    policy = RolePrecisionPolicy()
    spans = [RoleSpan(16, 32, "tool"), RoleSpan(0, 16, "system")]  # unsorted on purpose
    before = [(s.start, s.end, s.role) for s in spans]
    RoleKVCache(policy, spans)
    assert [(s.start, s.end, s.role) for s in spans] == before


def test_role_cache_honours_skip_layers():
    b, h, s, d = 1, 2, 32, 64
    policy = RolePrecisionPolicy(
        bits_per_role={"system": 2, "tool": 2}, default_bits=2, skip_layers=[0]
    )
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 32, "tool")]
    cache = RoleKVCache(policy, spans, requant_stride=1)
    key = torch.randn(b, h, s, d)
    value = torch.randn(b, h, s, d)
    out_k, out_v = cache.update(key.clone(), value.clone(), 0)
    # Layer 0 is skipped: returned tensors are bit-identical to the input.
    assert torch.equal(out_k, key)
    assert torch.equal(out_v, value)
