"""RoleKV: role-aware precision assignment for KV-cache compression.

Tokens in an agent conversation play different semantic roles (system
prompt, user turns, assistant replies, reasoning traces, tool outputs).
Where :class:`~onecomp.kvcomp.recipe.taperkv.TaperKVPolicy` grades precision
by token *age* and ThinKV by *thought type*, RoleKV grades it by *semantic
role*: easily re-computable or low-salience content (e.g. verbose tool
outputs) is stored at lower precision than the instructions that steer the
whole session.

Like ``TaperKVPolicy``, :class:`RolePrecisionPolicy` is a parameter carrier
consumed by its cache (:class:`~onecomp.kvcomp.cache.role_cache.RoleKVCache`,
live generation) and by the session checkpoint engine
(:mod:`onecomp.kvcomp.checkpoint`, offline ``.okv`` persistence); it does not
implement :meth:`~onecomp.kvcomp.KVPolicy.plan`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .._base import KVPolicy
from .._registry import register_kv_policy
from ..codec._packing import SUPPORTED_PACKED_BITS

KNOWN_ROLES = ("system", "user", "assistant", "tool", "reasoning")


@dataclass
class RoleSpan:
    """A contiguous token range annotated with a semantic role.

    Attributes:
        start: Start token index (inclusive).
        end: End token index (exclusive).
        role: Semantic role, one of ``KNOWN_ROLES``.
    """

    start: int
    end: int
    role: str

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span range [{self.start}, {self.end})")
        if self.role not in KNOWN_ROLES:
            raise ValueError(f"role must be one of {KNOWN_ROLES}, got {self.role!r}")


@register_kv_policy("rolekv")
@dataclass
class RolePrecisionPolicy(KVPolicy):
    """Mapping from semantic roles to KV quantization bit-widths.

    The defaults keep steering content (system prompt, user turns) at
    high precision while aggressively compressing tool outputs, which
    tend to be verbose and only locally relevant.

    Attributes:
        bits_per_role: Role name to bit-width mapping.
        default_bits: Bit-width used for roles absent from ``bits_per_role``.
        skip_layers: Layer indices left in full precision (outlier layers);
            honoured by the live :class:`RoleKVCache` path.
        sink_tokens: Leading "attention sink" tokens kept in full precision
            regardless of their role (StreamingLLM / KIVI remedy for
            outlier-heavy models); honoured by the live
            :class:`RoleKVCache` path.
    """

    bits_per_role: dict = field(
        default_factory=lambda: {
            "system": 8,
            "user": 8,
            "assistant": 4,
            "reasoning": 4,
            "tool": 2,
        }
    )
    default_bits: int = 4
    skip_layers: list = field(default_factory=list)
    sink_tokens: int = 4

    def __post_init__(self):
        super().__post_init__()
        for role, bits in self.bits_per_role.items():
            if bits not in SUPPORTED_PACKED_BITS:
                raise ValueError(
                    f"bits for role {role!r} must be one of {SUPPORTED_PACKED_BITS}, got {bits}"
                )
        if self.default_bits not in SUPPORTED_PACKED_BITS:
            raise ValueError(
                f"default_bits must be one of {SUPPORTED_PACKED_BITS}, got {self.default_bits}"
            )

    def bits_for(self, role: str) -> int:
        """Return the bit-width assigned to ``role``."""
        return self.bits_per_role.get(role, self.default_bits)

    def max_bits(self) -> int:
        """The most precise bit-width in the policy (incl. the default)."""
        return max(list(self.bits_per_role.values()) + [self.default_bits])

    def get_kv_config(self) -> dict:
        return {
            "name": "rolekv",
            "bits_per_role": dict(self.bits_per_role),
            "default_bits": self.default_bits,
            "skip_layers": list(self.skip_layers),
            "sink_tokens": self.sink_tokens,
        }


def validate_spans(spans: list, seq_length: int) -> list:
    """Check that spans are contiguous and cover ``[0, seq_length)``.

    Args:
        spans: Role spans to validate.
        seq_length: Sequence length that must be fully covered.

    Returns:
        The spans sorted by start index.

    Raises:
        ValueError: If the spans do not tile the sequence exactly.
    """
    ordered = sorted(spans, key=lambda s: s.start)
    if not ordered:
        raise ValueError("spans must not be empty")
    if ordered[0].start != 0:
        raise ValueError(f"spans must start at 0, got {ordered[0].start}")
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.end != cur.start:
            raise ValueError(f"spans must be contiguous: gap between {prev.end} and {cur.start}")
    if ordered[-1].end != seq_length:
        raise ValueError(
            f"spans must cover the full sequence: end {ordered[-1].end} != {seq_length}"
        )
    return ordered


def annotate_spans_from_chat(
    messages: list,
    tokenizer,
    chat_template_output_ids: list,
) -> list:
    """Approximate per-message token spans in a chat-templated sequence.

    The span boundaries are estimated by re-encoding the message prefix
    ``messages[:i + 1]`` with the tokenizer's chat template and taking the
    resulting length as the end of message ``i``.  This is an
    approximation: chat templates may insert generation prompts or
    separators whose attribution to a specific message is ambiguous, so
    boundaries can be off by a few template tokens.  Robust full coverage
    of the sequence is guaranteed (the final span is extended to the end
    of ``chat_template_output_ids``); exact per-message boundaries are not.

    Args:
        messages: Chat messages as ``{"role": ..., "content": ...}`` dicts.
        tokenizer: Tokenizer providing ``apply_chat_template``.
        chat_template_output_ids: Token ids of the fully templated
            conversation actually fed to the model.

    Returns:
        List of :class:`RoleSpan` covering ``[0, len(chat_template_output_ids))``.
    """
    total_length = len(chat_template_output_ids)
    spans = []
    prev_end = 0
    for i, message in enumerate(messages):
        role = message.get("role", "user")
        if role not in KNOWN_ROLES:
            role = "user"
        if i == len(messages) - 1:
            end = total_length
        else:
            prefix_ids = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            end = min(len(prefix_ids), total_length)
        if end > prev_end:
            spans.append(RoleSpan(start=prev_end, end=end, role=role))
            prev_end = end
    if prev_end < total_length:
        spans.append(RoleSpan(start=prev_end, end=total_length, role="assistant"))
    return spans


def role_token_bits(
    spans: list,
    total: int,
    policy: RolePrecisionPolicy,
    device,
    generated_role: str = "assistant",
) -> torch.Tensor:
    """Per-token bit-width vector for a role-annotated sequence.

    Bridges role spans to the :class:`~onecomp.kvcomp.cache._graded_base.
    GradedQuantCache` bit-schedule interface: positions covered by a span get
    that span's role bits; positions beyond the last span (tokens generated
    after annotation) get ``generated_role``'s bits; the leading
    ``policy.sink_tokens`` positions are kept in full precision (bits=16,
    which :class:`GradedQuantCache` leaves unquantized -- attention-sink
    protection, matching ``CompressedDynamicCache(sink_tokens=...)``).

    Args:
        spans: Role spans tiling the annotated prompt prefix (validated,
            sorted).
        total: Current sequence length of the layer cache.
        policy: Role-to-bits mapping.
        device: Device for the returned tensor.
        generated_role: Role assigned to positions past the last span.

    Returns:
        ``int64`` tensor of shape ``(total,)``.
    """
    bits = torch.full((total,), policy.bits_for(generated_role), dtype=torch.int64, device=device)
    for span in spans:
        start = min(span.start, total)
        end = min(span.end, total)
        if end > start:
            bits[start:end] = policy.bits_for(span.role)
    sink = min(policy.sink_tokens, total)
    if sink > 0:
        bits[:sink] = 16
    return bits
