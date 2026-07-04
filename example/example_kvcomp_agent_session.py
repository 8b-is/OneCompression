"""

Example: Agent session state with kvcomp -- KV-cache checkpoint / restore
and role-aware precision

Simulates an agent that is suspended while waiting for a slow tool call:
  1. Build an agent conversation (system prompt + user turn + tool output)
  2. Prefill the model and capture the KV cache
  3. Checkpoint the KV cache to disk (quantized, .okv container)
  4. Drop the live cache (the session is evicted from memory)
  5. Restore the cache from disk and continue generation seamlessly
  6. Role-aware variants: checkpoint tool-output tokens at lower precision,
     and generate live with RoleKVCache (per-role graded precision)

With bits=8 the restored greedy continuation matches the uninterrupted
run exactly; bits=4 halves the checkpoint size again with near-identical
continuations.

Copyright 2025-2026 Fujitsu Ltd.

"""

import dataclasses

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp import setup_logger
from onecomp.kvcomp import (
    KVCheckpointCodec,
    KVCheckpointConfig,
    RoleKVCache,
    RolePrecisionPolicy,
    annotate_spans_from_chat,
)

setup_logger()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STORE_DIR = "./kv_checkpoints"
MAX_NEW_TOKENS = 20

# -- 1. Load model and build an agent-style conversation --------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

messages = [
    {"role": "system", "content": "You are a weather assistant. Use tool results faithfully."},
    {"role": "user", "content": "What is the weather in Kawasaki right now?"},
    {
        "role": "tool",
        "content": '{"location": "Kawasaki", "temperature_c": 31, "humidity": 0.7,'
        ' "condition": "sunny"}',
    },
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
)
spans = annotate_spans_from_chat(messages, tokenizer, input_ids[0].tolist())
print(
    f"Prompt tokens: {input_ids.shape[1]}, role spans: {[(s.role, s.start, s.end) for s in spans]}"
)
gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None)

# -- 2. Prefill: process the prompt except its last token -------------
# The last prompt token stays unprocessed so generation can resume
# from the restored cache without re-running the prompt.
with torch.no_grad():
    prefill = model(input_ids[:, :-1], use_cache=True)

# Baseline: uninterrupted greedy continuation
baseline_ids = model.generate(input_ids, **gen_kwargs)
baseline_text = tokenizer.decode(baseline_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
print(f"\n[baseline]  {baseline_text}")

# -- 3. Checkpoint the KV cache (the agent now waits for a tool) ------
codec = KVCheckpointCodec(KVCheckpointConfig(bits=8, compress_backend="zlib", store_dir=STORE_DIR))
info = codec.checkpoint(prefill.past_key_values, session_id="agent-demo")
print(
    f"\nCheckpointed: {info.original_bytes} -> {info.compressed_bytes} bytes "
    f"(x{info.compression_ratio:.2f}) at {info.path}"
)

# -- 4. Evict the session, then restore and continue ------------------
del prefill
restored_cache = codec.restore("agent-demo")
restored_ids = model.generate(input_ids, past_key_values=restored_cache, **gen_kwargs)
restored_text = tokenizer.decode(restored_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
print(f"[restored]  {restored_text}")
print(f"Greedy match with baseline (bits=8): {restored_text == baseline_text}")

# -- 5. Role-aware checkpoint: tool output stored at fewer bits --------
# tool=4 bit is a robust operating point for small models; the default
# policy's tool=2 bit maximises compression but measurably hurts recall
# of tool facts (see the role-policy quality logs).
policy = RolePrecisionPolicy(
    bits_per_role={"system": 8, "user": 8, "assistant": 4, "reasoning": 4, "tool": 4}
)
role_codec = KVCheckpointCodec(
    KVCheckpointConfig(bits=4, compress_backend="zlib", store_dir=STORE_DIR, role_policy=policy)
)
with torch.no_grad():
    prefill = model(input_ids[:, :-1], use_cache=True)
# The cache covers all but the last prompt token; shrink the final span
# without mutating the annotated spans.
role_spans = spans[:-1] + [dataclasses.replace(spans[-1], end=input_ids.shape[1] - 1)]
role_info = role_codec.checkpoint(
    prefill.past_key_values, session_id="agent-demo-roles", spans=role_spans
)
print(
    f"\nRole-aware checkpoint: {role_info.original_bytes} -> {role_info.compressed_bytes} "
    f"bytes (x{role_info.compression_ratio:.2f})"
)
role_cache = role_codec.restore("agent-demo-roles")
role_ids = model.generate(input_ids, past_key_values=role_cache, **gen_kwargs)
role_text = tokenizer.decode(role_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
print(f"[role-aware] {role_text}")

# -- 6. Live role-graded generation with RoleKVCache -------------------
live_cache = RoleKVCache(policy, spans, requant_stride=1)
live_ids = model.generate(input_ids, past_key_values=live_cache, **gen_kwargs)
live_text = tokenizer.decode(live_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
print(f"\n[live rolekv] {live_text}")
print(f"Live whole-cache compression: x{live_cache.compression_ratio():.2f}")

print(f"\nStored checkpoints: {codec.list_checkpoints()}")
codec.delete("agent-demo")
codec.delete("agent-demo-roles")
