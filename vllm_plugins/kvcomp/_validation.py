"""
Dependency-free validation shared by KVComp's vLLM configuration layer.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
import operator
from typing import Any


def strict_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an integer without accepting lossy ``int(...)`` coercions.

    JSON integers, exactly integral finite JSON floats, and objects implementing
    Python's integer index protocol are accepted.  Booleans, fractional/non-
    finite floats, and numeric strings are rejected so malformed serialized
    model configuration cannot silently select a different compression recipe.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            result = int(value)
        else:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}")
    return result


__all__ = ["strict_integer"]
