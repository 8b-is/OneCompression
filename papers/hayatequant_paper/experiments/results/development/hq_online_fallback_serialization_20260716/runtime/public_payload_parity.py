"""Fresh-process online-vs-staging public payload bit-exact parity gate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[5]
PREFIX = "HQ_SERIALIZATION_PARITY_WORKER="
RATES = (2, 3, 4)


def _tensor_record(tensor) -> dict:
    import torch

    value = tensor.detach().cpu().contiguous()
    raw_value = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw_value.numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(value).all()),
    }


def _role_payload(payload: dict) -> dict:
    main = payload["main"]
    return {
        "mode": payload["mode"],
        "head_dim": int(payload["head_dim"]),
        "bits": int(main["bits"]),
        "code_bits": int(main.get("code_bits", int(main["bits"]) * int(main["block_size"]))),
        "block_size": int(main["block_size"]),
        "rotation": list(main["rot"]),
        "dist": main["dist"],
        "codebook_init": main["codebook_init"],
        "codes": _tensor_record(main["codes"]),
        "scale": _tensor_record(main["scale"]),
    }


def _worker(mode: str) -> dict:
    import torch

    from onecomp.kvcomp.codec import HayateQuantCodec
    from onecomp.kvcomp.codec.hayate.codec import _block_codebook_by_code_bits
    from onecomp.kvcomp.codec.hayate.recipes import build_hayate_hq

    if mode not in {"online", "artifact"}:
        raise RuntimeError(f"unknown worker mode {mode}")

    class SafeAdaptiveHQ(HayateQuantCodec):
        def _adaptive_code_bit_pairs(self, block_size: int):
            if int(block_size) != 2:
                raise ValueError("safe parity fixture requires b=2")
            total = int(self._pair_code_bits_total(2))
            rate = total // 4
            return [(2 * rate, 2 * rate), (2 * rate + 1, 2 * rate - 1)]

    result = {
        "mode": mode,
        "kvcomp_codebook_dir": os.environ.get("KVCOMP_CODEBOOK_DIR"),
        "rates": {},
        "codebooks": {},
    }
    for rate in RATES:
        torch.manual_seed(20260716 + rate)
        key = torch.randn(1, 2, 32, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key) * 0.75
        fixed = build_hayate_hq(rate, encoder="exact")
        fixed_encoded = fixed.encode(key, value)
        fixed_decoded = fixed.decode(fixed_encoded)
        adaptive = SafeAdaptiveHQ(
            bits=rate,
            value_bits=rate,
            avg_bits=float(rate),
            block_size=2,
            key_bit_fraction=0.5,
            adaptive_bit_split="online_stats",
            adaptive_stats_min_tokens=16,
            adaptive_bit_margin=1,
            key_error_weight=4.0,
            value_error_weight=1.0,
            max_codebook_bits=9,
            dist="gaussian",
            codebook_init="auto",
            rotation_mode="auto",
            encoder="exact",
            key_scale_mode="nc",
            value_scale_mode="ls_iter",
        )
        adaptive_encoded = adaptive.encode(key, value)
        adaptive_decoded = adaptive.decode(adaptive_encoded)
        split = adaptive_encoded.meta["adaptive_bit_split"]
        result["rates"][str(rate)] = {
            "input_key": _tensor_record(key),
            "input_value": _tensor_record(value),
            "fixed": {
                "key": _role_payload(fixed_encoded.key_q),
                "value": _role_payload(fixed_encoded.value_q),
                "stored_bits": int(fixed_encoded.meta["stored_bits"]),
                "decoded_key": _tensor_record(fixed_decoded[0]),
                "decoded_value": _tensor_record(fixed_decoded[1]),
            },
            "adaptive": {
                "key": _role_payload(adaptive_encoded.key_q),
                "value": _role_payload(adaptive_encoded.value_q),
                "stored_bits": int(adaptive_encoded.meta["stored_bits"]),
                "candidate_pairs": [list(pair) for pair in split["candidates"]],
                "selected_pair_index": int(split["selected_pair_index"]),
                "selected_pair": [
                    int(split["key_code_bits_per_block"]),
                    int(split["value_code_bits_per_block"]),
                ],
                "selector_bits": int(split["selector_bits"]),
                "selector_incremental_stored_bits": int(
                    split["selector_incremental_stored_bits"]
                ),
                "decoded_key": _tensor_record(adaptive_decoded[0]),
                "decoded_value": _tensor_record(adaptive_decoded[1]),
            },
        }
        for code_bits in (2 * rate - 1, 2 * rate, 2 * rate + 1):
            cb = _block_codebook_by_code_bits(code_bits, 2, "gaussian", 128, "auto")
            result["codebooks"][f"b2_k{1 << code_bits}"] = _tensor_record(cb)
    return result


def _run_worker(mode: str) -> dict:
    env = dict(os.environ)
    if mode == "artifact":
        env["KVCOMP_CODEBOOK_DIR"] = str(ROOT / "artifacts")
    else:
        env.pop("KVCOMP_CODEBOOK_DIR", None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} worker failed: stdout={completed.stdout[-2000:]} "
            f"stderr={completed.stderr[-4000:]}"
        )
    line = next(
        (row for row in reversed(completed.stdout.splitlines()) if row.startswith(PREFIX)),
        None,
    )
    if line is None:
        raise RuntimeError(f"{mode} worker emitted no parity record")
    return json.loads(line[len(PREFIX) :])


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(PREFIX + json.dumps(_worker(sys.argv[2]), sort_keys=True), flush=True)
        return
    online = _run_worker("online")
    artifact = _run_worker("artifact")
    online_comparable = {key: value for key, value in online.items() if key not in {"mode", "kvcomp_codebook_dir"}}
    artifact_comparable = {
        key: value for key, value in artifact.items() if key not in {"mode", "kvcomp_codebook_dir"}
    }
    bit_exact = online_comparable == artifact_comparable
    document = {
        "schema": "hayatequant.current_online_serialization.public_payload_parity.v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_processes": True,
        "online_artifact_bit_exact": bit_exact,
        "checks": {
            "all_codebooks": online["codebooks"] == artifact["codebooks"],
            "all_fixed_rates": all(
                online["rates"][str(rate)]["fixed"]
                == artifact["rates"][str(rate)]["fixed"]
                for rate in RATES
            ),
            "all_adaptive_rates": all(
                online["rates"][str(rate)]["adaptive"]
                == artifact["rates"][str(rate)]["adaptive"]
                for rate in RATES
            ),
            "inputs_identical": all(
                online["rates"][str(rate)]["input_key"]
                == artifact["rates"][str(rate)]["input_key"]
                and online["rates"][str(rate)]["input_value"]
                == artifact["rates"][str(rate)]["input_value"]
                for rate in RATES
            ),
        },
        "online": online,
        "artifact": artifact,
    }
    output = ROOT / "public_payload_parity.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)
    if not bit_exact or not all(document["checks"].values()):
        raise RuntimeError("online and serialized-artifact public payloads differ")


if __name__ == "__main__":
    main()
