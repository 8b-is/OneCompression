# Rate-2 native eligibility preregistration

Frozen while Slurm quality array 69327 was still running and before its global
candidate rank was computed.  This is a separate development gate, not formal
speed evidence and not a license to alter the fixed candidate set.

All three preregistered b=4 K10/V6 exact candidates and the current b=2 K5/V3
HQ control receive the existing packed-native parity smoke.  The smoke checks
public/native code and signed-fp16-scale parity, pack/unpack, untouched padding,
dequantization, packed attention, one-token/prefill identity, finite output,
and logical/physical rate.

Native timing compares each b=4 candidate to current HQ by fresh-process ABBA:
current, candidate, candidate, current.  The same B200, random tensors, physical
68-byte slot, 4096 tokens, 8 KV heads, 32 query heads, head dimension 128,
bfloat16, 16-token pages, and 32 decode splits are used.  Compilation and five
warmups are excluded; each process supplies 20 CUDA-event samples for prefill
store, packed decode attention, and store-one-plus-decode.  No upstream 2-bit
TurboQuant native preset exists, so this gate makes no invented rate-2 TQ
native comparison.  Existing public-API TQ timing remains separately reported.

Every failure and regression is retained.  A b=4 quality win is still
ineligible for canonical HQ if parity fails or if the generic b=4 native path
has a fatal latency regression relative to current b=2 HQ.
