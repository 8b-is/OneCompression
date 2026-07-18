"""Regression tests for fail-closed paper aggregation of synthetic retrieval."""

from papers.aggregate_results import audited_needle, audited_synthetic_multikey


def _grid_row(*, hit=True):
    return {
        "requested_filler_tokens": 1024,
        "requested_depth": 0.5,
        "prompt_seed": 1,
        "rendered_prompt_tokens": 1051,
        "reserved_prompt_plus_generation_tokens": 1067,
        "model_context_capacity": 4096,
        "prompt_sha256": "a" * 64,
        "gold": "1234",
        "prediction": "1234",
        "hit": hit,
        "cache_accounting": {"compression_aggregation": "fullkv_no_cache"},
    }


def test_needle_aggregate_rejects_legacy_or_inconsistent_trial_ledger():
    needle = {
        "benchmark": "deterministic synthetic needle-in-a-haystack v2",
        "protocol_version": 2,
        "official_reference": False,
        "n": 2,
        "hits": 1,
        "accuracy": 0.5,
        "grid": [_grid_row(hit=True), _grid_row(hit=False)],
    }
    assert audited_needle({"needle": needle}) == needle

    legacy = dict(needle, protocol_version=1)
    assert audited_needle({"needle": legacy}) is None
    inconsistent = dict(needle, hits=2)
    assert audited_needle({"needle": inconsistent}) is None
    malformed_hash = dict(
        needle,
        grid=[dict(_grid_row(), prompt_sha256="not-a-hash"), _grid_row(hit=False)],
    )
    assert audited_needle({"needle": malformed_hash}) is None
    legacy_accounting = dict(
        needle,
        grid=[
            dict(
                _grid_row(),
                cache_accounting={
                    "compression_aggregation": "legacy_cache_ratio_only",
                    "compression_ratio": 3.0,
                },
            ),
            _grid_row(hit=False),
        ],
    )
    assert audited_needle({"needle": legacy_accounting}) is None


def test_multikey_aggregate_is_never_accepted_as_ruler():
    result = {
        "benchmark": "deterministic synthetic multi-key retrieval v2",
        "protocol_version": 2,
        "official_reference": False,
        "n": 1,
        "hits": 1,
        "grid": [_grid_row()],
    }
    assert audited_synthetic_multikey({"synthetic_multikey": result}) == result
    assert audited_synthetic_multikey({"ruler": result}) is None
