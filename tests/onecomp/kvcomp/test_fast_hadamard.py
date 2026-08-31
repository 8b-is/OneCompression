"""Parity tests for the Dao-AILab fast-Hadamard CUDA kernel dispatch.

The kernel path in :func:`onecomp.kvcomp.codec.rotation.walsh_hadamard_transform`
must be numerically interchangeable with the pure-PyTorch butterfly (and with
the folded Hadamard GEMM), otherwise the TurboQuant reduction of HayateQuant
would silently stop being bit-faithful.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp.kvcomp.codec import rotation as R

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_fht = pytest.mark.skipif(
    R._fht_cuda is None, reason="fast_hadamard_transform not installed"
)


@requires_cuda
@requires_fht
@pytest.mark.parametrize("d", [64, 128, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fast_hadamard_parity(d, dtype):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 33, d, device="cuda", dtype=dtype)
    y_kernel = R.walsh_hadamard_transform(x)
    y_ref = R._wht_butterfly(x.float())
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    assert y_kernel.dtype == x.dtype
    assert torch.allclose(y_kernel.float(), y_ref, atol=tol, rtol=tol)


@requires_cuda
@requires_fht
def test_fast_hadamard_involution():
    torch.manual_seed(1)
    x = torch.randn(16, 128, device="cuda")
    y = R.walsh_hadamard_transform(R.walsh_hadamard_transform(x))
    assert torch.allclose(y, x, atol=1e-5)


@requires_cuda
@requires_fht
def test_fast_hadamard_matches_folded_gemm():
    """Kernel path == the folded signed-Hadamard GEMM used by HayateQuant."""
    torch.manual_seed(2)
    d, seed = 128, 7
    x = torch.randn(64, d, device="cuda")
    sign = R.random_sign(d, seed, device=x.device, dtype=torch.float32)
    y_kernel = R.rotate(x, sign)
    hs = R.signed_hadamard_matrix(d, seed, str(x.device), torch.float32)
    y_gemm = x @ hs
    assert torch.allclose(y_kernel, y_gemm, atol=1e-4)


def test_wht_cpu_fallback_unchanged():
    torch.manual_seed(3)
    x = torch.randn(8, 64)
    y = R.walsh_hadamard_transform(x)
    assert torch.allclose(R.walsh_hadamard_transform(y), x, atol=1e-5)
    h = R.hadamard_matrix(64, "cpu", torch.float32)
    assert torch.allclose(y, x @ h, atol=1e-5)
