"""Single-process vLLM benchmark for OneComp KV-cache methods.

Each invocation loads one model and one method.  Keeping method/rate pairs in
fresh processes avoids process-global backend registry and CUDA allocator
state leaking between measurements.  The script writes both a compact JSON
result and backend-emitted JSONL evidence proving which data plane executed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import sys
import tempfile
import time

from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    run_code_commit,
    run_code_revision,
    validate_model_provenance,
)

METHODS = ("fullkv", "turboquant", "hayatequant", "taperkv")
VLLM_IPC_PATH_MAX_LEN = 107
VLLM_IPC_NAME_LENGTH = 36


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num_hidden_layers(config: dict) -> int:
    """Resolve the dense-transformer layer count from a local HF config."""

    candidates = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.insert(0, text_config)
    for candidate in candidates:
        value = candidate.get("num_hidden_layers")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError("model config does not declare a positive num_hidden_layers")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _kv_recipe(method: str, rate: int) -> dict | None:
    from onecomp.kvcomp import KVCompressionConfig, TaperKVPolicy
    from onecomp.kvcomp.codec import TurboQuantCodec
    from onecomp.kvcomp.codec.hayate.recipes import build_hayatequant

    if method == "fullkv":
        return None
    if method == "turboquant":
        codec = TurboQuantCodec(bits=rate, dist="beta", rotation_mode="hadamard")
        return KVCompressionConfig(codec=codec, residual_window=0).to_dict()
    if method == "hayatequant":
        codec = build_hayatequant(rate)
        return KVCompressionConfig(codec=codec, residual_window=0).to_dict()
    if method == "taperkv":
        return KVCompressionConfig(policy=TaperKVPolicy(), residual_window=0).to_dict()
    raise ValueError(f"unknown method {method!r}")


@contextmanager
def _model_overlay(model: Path, recipe: dict | None, parent: Path):
    """Create a temporary symlink view with a non-destructive config override."""

    if recipe is None:
        yield model
        return
    source_config = model / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"local model has no config.json: {source_config}")
    overlay = Path(tempfile.mkdtemp(prefix="kvcomp-model-", dir=parent))
    try:
        for source in model.iterdir():
            if source.name == "config.json":
                continue
            os.symlink(
                source.resolve(), overlay / source.name, target_is_directory=source.is_dir()
            )
        config = json.loads(source_config.read_text(encoding="utf-8"))
        config["kv_compression_config"] = recipe
        (overlay / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield overlay
    finally:
        shutil.rmtree(overlay)


def _prompt_ids(tokenizer, length: int) -> list[int]:
    seed_text = (
        "The history of cartography reflects careful observation, measurement, "
        "and the gradual exchange of knowledge across many cultures. "
    )
    text = seed_text
    ids = tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < length:
        text += seed_text
        ids = tokenizer.encode(text, add_special_tokens=True)
    return ids[:length]


def _read_evidence(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_layer_coverage(
    method: str,
    event: str,
    rows: list[dict],
    expected_layer_count: int | None,
) -> None:
    if expected_layer_count is None:
        return
    expected = set(range(expected_layer_count))
    observed = {row.get("layer_index") for row in rows if row.get("event") == event}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(index for index in observed - expected if index is not None)
        unresolved = None in observed
        raise RuntimeError(
            f"{method} did not cover every model layer for {event}: "
            f"missing={missing}, unexpected={unexpected}, unresolved={unresolved}"
        )


def _validate_runtime_evidence(
    method: str,
    evidence: list[dict],
    expected_layer_count: int | None = None,
    rate: int | None = None,
) -> None:
    """Fail closed unless the requested custom data plane actually ran."""

    expected_schema = {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1}
    if any(row.get("schema") != expected_schema for row in evidence):
        raise RuntimeError("custom-backend runtime evidence has an invalid schema")
    if method == "turboquant" and rate == 2:
        required = {
            "packed_turboquant2_store_launched",
            "packed_turboquant2_decode_attention_launched",
        }
        events = {row.get("event") for row in evidence}
        missing = sorted(required - events)
        if missing:
            raise RuntimeError(f"turboquant rate 2 runtime evidence is missing: {missing}")
        for row in evidence:
            if row.get("event") not in required:
                continue
            if row.get("implementation") != "native_packed_turboquant2":
                raise RuntimeError("turboquant rate 2 did not use its packed implementation")
            if row.get("rate") != 2 or row.get("key_bits") != 2 or row.get("value_bits") != 2:
                raise RuntimeError("turboquant rate 2 runtime recipe is inconsistent")
            if row.get("cache_dtype") != "uint8":
                raise RuntimeError("turboquant rate 2 cache is not packed uint8 storage")
            if not str(row.get("cache_device", "")).startswith("cuda"):
                raise RuntimeError("turboquant rate 2 physical cache is not on CUDA")
        for event in sorted(required):
            _validate_layer_coverage("turboquant rate 2", event, evidence, expected_layer_count)
        return
    if method in {"fullkv", "turboquant"}:
        if evidence:
            raise RuntimeError(f"{method} unexpectedly emitted custom-backend evidence")
        return
    if not evidence:
        raise RuntimeError(f"{method} produced no custom-backend runtime evidence")

    events = {row.get("event") for row in evidence}
    if method == "hayatequant":
        required = {
            "packed_hayatequant_store_launched",
            "packed_hayatequant_decode_attention_launched",
        }
        missing = sorted(required - events)
        if missing:
            raise RuntimeError(f"hayatequant runtime evidence is missing events: {missing}")
        for row in evidence:
            if row.get("event") not in required:
                continue
            if row.get("implementation") != "native_packed_hayatequant":
                raise RuntimeError("hayatequant did not use its native packed implementation")
            if (
                row.get("event") == "packed_hayatequant_store_launched"
                and row.get("store_encoder") != "triton_fused_b2_k_nc_v_ls_iter_encode_pack"
            ):
                raise RuntimeError("hayatequant did not use its fused CUDA store encoder")
            if row.get("cache_dtype") != "uint8":
                raise RuntimeError("hayatequant physical cache is not packed uint8 storage")
            if not str(row.get("cache_device", "")).startswith("cuda"):
                raise RuntimeError("hayatequant physical cache is not on CUDA")
        for event in sorted(required):
            _validate_layer_coverage("hayatequant", event, evidence, expected_layer_count)
        return
    if method == "taperkv":
        required_event = "gpu_paged_cache_regraded"
        rows = [row for row in evidence if row.get("event") == required_event]
        if not rows:
            raise RuntimeError(f"taperkv runtime evidence is missing event: {required_event}")
        for row in rows:
            if row.get("backend") != "vllm_plugins.kvcomp.taper_backend.TaperKVBackend":
                raise RuntimeError("taperkv did not use its GPU semantic backend")
            if not str(row.get("physical_cache_device", "")).startswith("cuda"):
                raise RuntimeError("taperkv physical cache is not on CUDA")
            if row.get("new_token_cache_update") != (
                "triton_fused_affine_quantize_and_paged_scatter"
            ):
                raise RuntimeError("taperkv did not use its fused quantize-and-store update")
        _validate_layer_coverage("taperkv", required_event, rows, expected_layer_count)
        return
    raise RuntimeError(f"unknown method for runtime evidence validation: {method!r}")


def _ipc_tmpdir_fits(path: Path) -> bool:
    """Return whether vLLM can append its UUID Unix-socket name."""

    socket_path = path / ("x" * VLLM_IPC_NAME_LENGTH)
    return len(os.fsencode(socket_path)) <= VLLM_IPC_PATH_MAX_LEN


def _compact_runtime_tmpdir(output: Path) -> Path:
    """Choose a writable output ancestor with room for vLLM IPC sockets."""

    resolved = output.resolve()
    identity = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:8]
    for ancestor in resolved.parents:
        candidate = ancestor / ".onecomp-tmp" / identity
        if not _ipc_tmpdir_fits(candidate):
            continue
        if not os.access(ancestor, os.W_OK | os.X_OK):
            continue
        return candidate
    raise RuntimeError(
        "cannot place vLLM IPC sockets below a writable output ancestor; "
        "use a shorter output path or provide a shorter writable TMPDIR"
    )


def _runtime_tmpdir(output: Path) -> Path:
    preferred = os.environ.get("TMPDIR")
    if preferred:
        candidate = Path(preferred).expanduser().resolve()
        if _ipc_tmpdir_fits(candidate):
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            return candidate
    candidate = _compact_runtime_tmpdir(output)
    candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"unsafe runtime TMPDIR: {candidate}")
    return candidate


def _configure_runtime_environment(output: Path) -> Path:
    """Confine generated caches and load only the benchmark's vLLM plugin."""

    runtime_root = (output.parent / "runtime" / output.stem).resolve()
    locations = {
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "VLLM_CACHE_ROOT": runtime_root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "FLASHINFER_WORKSPACE_BASE": runtime_root / "flashinfer",
        "HF_HOME": runtime_root / "huggingface",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
    }
    runtime_root.mkdir(parents=True, exist_ok=True)
    for variable, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    # Calling a virtual environment's Python by absolute path does not add its
    # sibling console scripts to PATH.  vLLM/FlashInfer launches its installed
    # ``ninja`` executable by name during JIT compilation.
    interpreter_bin = str(Path(sys.executable).absolute().parent)
    path_entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and entry != interpreter_bin
    ]
    os.environ["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])
    os.environ["TMPDIR"] = str(_runtime_tmpdir(output))
    # The project can expose unrelated vLLM plugins.  Loading only kvcomp
    # prevents their optional version-specific dependencies from affecting a
    # KV-cache benchmark, and the spawned engine inherits this selection.
    os.environ["VLLM_PLUGINS"] = "kvcomp"
    # Importing a local compression stack can initialize CUDA before vLLM
    # creates its engine process.  Linux's default ``fork`` context cannot
    # safely inherit that CUDA runtime, so select the supported clean process
    # start explicitly and before importing vLLM.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # This benchmark compares KV-cache kernels under the same dense-model
    # execution path.  Disable optional FP8/MoE DeepGEMM so a Blackwell host
    # without that separate extension does not fail during unrelated warmup.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    return runtime_root


def _engine_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.prompt_tokens + args.max_tokens,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        # vLLM's offline LLM default disables RequestStateStats, which would
        # make decode-only throughput unverifiable even though generation ran.
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
    }
    if args.method == "turboquant":
        from vllm_plugins.kvcomp.native_turboquant import turboquant_engine_kwargs

        kwargs.update(turboquant_engine_kwargs(args.rate))
        os.environ["KVCOMP_VLLM_TURBOQUANT_MODE"] = "native"
    return kwargs


def _decode_measurement(metrics, generated_tokens: int) -> tuple[int, float | None, float | None]:
    decode_tokens = max(0, generated_tokens - 1)
    if (
        metrics is None
        or metrics.first_token_ts <= 0
        or metrics.last_token_ts < metrics.first_token_ts
    ):
        return decode_tokens, None, None
    decode_seconds = float(metrics.last_token_ts - metrics.first_token_ts)
    if decode_seconds <= 0:
        return decode_tokens, decode_seconds, None
    return decode_tokens, decode_seconds, decode_tokens / decode_seconds


def run(args: argparse.Namespace) -> dict:
    model = args.model.resolve()
    if not model.is_dir():
        raise FileNotFoundError("vLLM benchmark requires a local model directory")
    model_config = model / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(f"local model has no config.json: {model_config}")
    model_config_payload = json.loads(model_config.read_text(encoding="utf-8"))
    expected_layer_count = _num_hidden_layers(model_config_payload)
    model_config_sha256 = _sha256(model_config)
    model_attestation = validate_model_provenance(model)
    model_checkpoint_fingerprint = (
        model_attestation.get("checkpoint", {}).get("fingerprint_sha256")
        if model_attestation is not None
        else None
    )
    code_commit = run_code_commit()
    code_revision = run_code_revision()
    run_provenance = load_run_provenance()
    expected_rate = args.rate if args.method in {"turboquant", "hayatequant"} else None
    effective_eager = args.enforce_eager
    compilation_mode = "disabled_by_enforce_eager" if effective_eager else "vllm_compile"
    cuda_graph_mode = "none" if effective_eager else "vllm_default"
    layer_contract = {
        "expected_model_layers": expected_layer_count,
        "explicit_skip_layers": [],
        "require_all_layers": args.method != "fullkv",
        "vllm_implicit_turboquant_boundary_skip_disabled": (
            True if args.method in {"turboquant", "hayatequant"} else None
        ),
    }
    if args.output.exists() and not args.overwrite:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 3,
            "method": args.method,
            "rate": expected_rate,
            "model": str(model),
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": effective_eager,
            "compilation_mode": compilation_mode,
            "cuda_graph_mode": cuda_graph_mode,
            "request_metrics_enabled": True,
            "vllm_deep_gemm_enabled": False,
            "model_config_sha256": model_config_sha256,
            "model_checkpoint_fingerprint": model_checkpoint_fingerprint,
            "code_revision": code_revision,
            "compression_layer_contract": layer_contract,
        }
        mismatch = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatch:
            raise FileExistsError(
                f"existing vLLM result does not match this invocation: {mismatch}"
            )
        return existing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output.with_suffix(".runtime.jsonl").resolve()
    if evidence_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"runtime evidence already exists: {evidence_path}")
        evidence_path.unlink()
    os.environ["KVCOMP_VLLM_RUNTIME_EVIDENCE"] = str(evidence_path)
    runtime_root = _configure_runtime_environment(args.output)

    recipe = _kv_recipe(args.method, args.rate)

    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=args.trust_remote_code)
    prompt_ids = _prompt_ids(tokenizer, args.prompt_tokens)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=args.seed,
    )

    with _model_overlay(model, recipe, args.output.parent) as engine_model:
        started = time.perf_counter()
        llm = LLM(
            model=str(engine_model),
            tokenizer=str(model),
            **_engine_kwargs(args),
        )
        initialization_seconds = time.perf_counter() - started
        prompt = {"prompt_token_ids": prompt_ids}

        # Warm kernels, codebooks, and CUDA graphs before measured repeats.
        llm.generate([prompt], sampling, use_tqdm=False)
        samples = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            output = outputs[0]
            generated_token_ids = list(output.outputs[0].token_ids)
            generated = len(generated_token_ids)
            decode_tokens, decode_seconds, decode_rate = _decode_measurement(
                output.metrics,
                generated,
            )
            samples.append(
                {
                    "repeat": repeat,
                    "generated_tokens": generated,
                    "e2e_seconds": elapsed,
                    "e2e_tokens_per_second": generated / elapsed,
                    "decode_tokens": decode_tokens,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": decode_rate,
                    "finish_reason": output.outputs[0].finish_reason,
                    "generated_token_ids_sha256": hashlib.sha256(
                        json.dumps(
                            generated_token_ids,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest(),
                }
            )

    decode_rates = [
        row["decode_tokens_per_second"]
        for row in samples
        if row["decode_tokens_per_second"] is not None
    ]
    if not decode_rates:
        raise RuntimeError(
            "vLLM returned no decode timing; RequestStateStats must be enabled "
            "for a formal speed result"
        )
    evidence = _read_evidence(evidence_path)
    _validate_runtime_evidence(args.method, evidence, expected_layer_count, args.rate)
    hayatequant_projection_fusion = None
    if args.method == "hayatequant":
        decode_evidence = [
            row
            for row in evidence
            if row.get("event") == "packed_hayatequant_decode_attention_launched"
        ]
        hayatequant_projection_fusion = {
            "output_projection_layer_indices": sorted(
                row["layer_index"]
                for row in decode_evidence
                if row.get("output_projection_fused") is True
            ),
            "value_projection_layer_indices": sorted(
                row["layer_index"]
                for row in decode_evidence
                if row.get("value_projection_fused") is True
            ),
        }
    if args.method in {"turboquant", "hayatequant"}:
        from vllm_plugins.kvcomp.native_turboquant import (
            vllm_implicit_turboquant_boundary_skip_disabled,
        )

        if not vllm_implicit_turboquant_boundary_skip_disabled():
            raise RuntimeError("vLLM implicit TurboQuant boundary skips remain enabled")
    result = {
        "schema_version": 3,
        "method": args.method,
        "rate": expected_rate,
        "model": str(model),
        "model_config_sha256": model_config_sha256,
        "model_checkpoint_fingerprint": model_checkpoint_fingerprint,
        "code_commit": code_commit,
        "code_revision": code_revision,
        "run_provenance": [run_provenance] if run_provenance is not None else [],
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": effective_eager,
        "compilation_mode": compilation_mode,
        "cuda_graph_mode": cuda_graph_mode,
        "request_metrics_enabled": True,
        "vllm_deep_gemm_enabled": False,
        "compression_layer_contract": layer_contract,
        "compressed_layer_indices": (
            list(range(expected_layer_count)) if args.method != "fullkv" else []
        ),
        "physical_cache_contract": (
            "onecomp_native_packed_turboquant2"
            if args.method == "turboquant" and args.rate == 2
            else {
                "fullkv": "model_dtype",
                "turboquant": "vllm_native_packed",
                "hayatequant": "onecomp_native_packed",
                "taperkv": "model_dtype_semantic_regrading",
            }[args.method]
        ),
        "hayatequant_projection_fusion": hayatequant_projection_fusion,
        "accelerator": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "vllm": str(vllm.__version__),
        },
        "initialization_seconds": initialization_seconds,
        "median_e2e_tokens_per_second": median(row["e2e_tokens_per_second"] for row in samples),
        "median_decode_tokens_per_second": median(decode_rates) if decode_rates else None,
        "samples": samples,
        "runtime_evidence_file": str(evidence_path),
        "runtime_evidence": evidence,
        "runtime_cache_root": str(runtime_root),
        "runtime_tmpdir": os.environ["TMPDIR"],
        "kv_compression_config": recipe,
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--rate", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.max_tokens < 2 or args.repeats < 1:
        parser.error("prompt tokens/repeats must be positive and --max-tokens must be >= 2")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
