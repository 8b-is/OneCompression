"""End-to-end GPU verification: HayateQuant (Hadamard, block_size=1) through
the OneComp vLLM plugin drives vLLM's native TurboQuant KV cache.

Steps:
1. Build a temp model dir (symlinked weights) whose config.json carries the
   ``kv_compression_config`` a Runner-saved HayateQuant model would have.
2. Launch a real ``vllm.LLM`` on it -- the ``onecomp_kvcomp`` general plugin
   (entry point) must auto-set ``kv_cache_dtype='turboquant_4bit_nc'``.
3. Assert the engine's resolved cache dtype and generate text with both the
   compressed and uncompressed engines for a sanity comparison.
"""
import json
import os
import shutil
import sys
import tempfile

MODEL = "/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-1.5B"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from onecomp.kvcomp.codec.hayate.codec import HayateQuantCodec  # noqa: E402


def make_model_dir(codec_cfg: dict) -> str:
    d = tempfile.mkdtemp(prefix="hayate_vllm_")
    for name in os.listdir(MODEL):
        os.symlink(os.path.join(MODEL, name), os.path.join(d, name))
    os.remove(os.path.join(d, "config.json"))
    with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["kv_compression_config"] = {"codec": codec_cfg}
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return d


def run_engine(model_path: str, tag: str):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=2048, gpu_memory_utilization=0.55,
              enforce_eager=True)
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    print(f"[{tag}] resolved cache_dtype = {cache_cfg.cache_dtype}")
    out = llm.generate(
        ["The capital of France is", "In one sentence, what is a Hadamard matrix?"],
        SamplingParams(temperature=0.0, max_tokens=48),
    )
    for o in out:
        print(f"[{tag}] {o.prompt!r} -> {o.outputs[0].text[:160]!r}")
    dtype = str(cache_cfg.cache_dtype)
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return dtype


def main():
    codec = HayateQuantCodec(bits=4, block_size=1, rotation_mode="hadamard_gemm")
    model_dir = make_model_dir(codec.get_kv_config())
    print("model_dir:", model_dir)

    dtype = run_engine(model_dir, "hayatequant-hadamard-bs1")
    assert dtype == "turboquant_4bit_nc", f"expected turboquant_4bit_nc, got {dtype}"
    print("HADAMARD_NATIVE_KV_OK")

    base = run_engine(MODEL, "baseline-fp16")
    assert base == "auto", f"baseline unexpectedly compressed: {base}"
    print("BASELINE_OK")
    shutil.rmtree(model_dir, ignore_errors=True)

    # Case 3: default HayateQuant (auto -> block-VQ, block_size=2 at 4-bit)
    # with the opt-in scalar fallback env var: the plugin must serve the
    # model compressed with the nearest scalar turboquant_* preset instead of
    # refusing native support (uncompressed cache).
    codec_bvq = HayateQuantCodec(bits=4)
    model_dir_bvq = make_model_dir(codec_bvq.get_kv_config())
    os.environ["KVCOMP_VLLM_SCALAR_FALLBACK"] = "1"
    try:
        dtype_fb = run_engine(model_dir_bvq, "hayatequant-blockvq-scalar-fallback")
    finally:
        del os.environ["KVCOMP_VLLM_SCALAR_FALLBACK"]
    assert dtype_fb == "turboquant_4bit_nc", f"expected turboquant_4bit_nc, got {dtype_fb}"
    print("BLOCKVQ_SCALAR_FALLBACK_OK")
    shutil.rmtree(model_dir_bvq, ignore_errors=True)
    print("ALL_E2E_CHECKS_PASSED")


if __name__ == "__main__":
    main()
