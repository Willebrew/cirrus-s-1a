#!/usr/bin/env python3
"""DGX Spark microbenchmarks for cirrus-s-1a (roadmap Phase 1a).

Measures every number the roofline model currently estimates, replays the
model's exact decode-step weight stream with random NVFP4 weights, then
re-runs the speed prediction with the measured hardware values.

    pip install torch ninja        # CUDA build of torch with sm_121 support
    python scripts/bench_spark.py  # ~3-5 minutes, writes results/spark-*.json

Run it on an otherwise idle GPU: everything here is bandwidth-sensitive.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cirrus.config import get_config  # noqa: E402
from cirrus.perf.hardware import DGX_SPARK  # noqa: E402
from cirrus.perf.roofline import InferencePlan, decode_step  # noqa: E402
from cirrus.perf.shapes import decode_weight_matrices  # noqa: E402

MB = 1 << 20
RESULTS: dict = {"sections": {}, "errors": {}}


def log(msg: str) -> None:
    print(msg, flush=True)


def section(name: str):
    """Run a benchmark section; record its result or its error, never abort."""

    def wrap(fn):
        def run(*a, **kw):
            log(f"\n== {name} ==")
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                RESULTS["sections"][name] = out
                log(f"   ({time.time() - t0:.1f}s)")
                return out
            except Exception as e:  # keep going; report at the end
                RESULTS["errors"][name] = f"{type(e).__name__}: {e}"
                log(f"   FAILED: {e}")
                traceback.print_exc(limit=3)
                return None

        return run

    return wrap


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def cuda_time_ms(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median wall time of fn() on the GPU, in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


# ------------------------------------------------------------------ sections
@section("environment")
def environment() -> dict:
    p = torch.cuda.get_device_properties(0)
    env = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": p.name,
        "capability": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "total_mem_gb": round(p.total_memory / 1e9, 1),
        "l2_bytes": getattr(p, "L2_cache_size", None),
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv,noheader"),
        "nvcc": sh("nvcc --version | tail -2"),
        "dgx_os": sh("cat /etc/dgx-release 2>/dev/null | head -5"),
        "gpu_processes": sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
    }
    for mod in ("triton", "fla"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    for k, v in env.items():
        log(f"   {k}: {v}")
    others = [l for l in env["gpu_processes"].splitlines() if l.strip() and str(os.getpid()) not in l]
    if others:
        log("   WARNING: other GPU processes are running; bandwidth numbers will read low:")
        for l in others:
            log(f"     {l}")
    return env


def build_ext():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    src = (ROOT / "cirrus/perf/bench_kernels.cu").read_text()
    decls = """
double time_read(torch::Tensor buf, int64_t iters, int64_t blocks, int64_t threads);
double time_coop_barrier(int64_t n, int64_t blocks, int64_t threads);
double time_flag_barrier(int64_t n, int64_t blocks, int64_t threads);
torch::Tensor nvfp4_gemv(torch::Tensor W, torch::Tensor S, torch::Tensor X, double gscale);
torch::Tensor nvfp4_mma_gemv(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t N, double gscale, torch::Tensor Y);
void run_stream(torch::Tensor descs, torch::Tensor stage_first, int64_t n_stages, int64_t M, int64_t kmax);
double time_stream(torch::Tensor descs, torch::Tensor stage_first, int64_t n_stages, int64_t M, int64_t iters, int64_t kmax);
void run_stream2(torch::Tensor descs, torch::Tensor stage_first, torch::Tensor tasks, torch::Tensor stf, int64_t n_stages, int64_t M, int64_t kmax, int64_t policy);
double time_stream2(torch::Tensor descs, torch::Tensor stage_first, torch::Tensor tasks, torch::Tensor stf, int64_t n_stages, int64_t M, int64_t iters, int64_t kmax, int64_t policy);
double time_read_rot(torch::Tensor buf, int64_t iters, int64_t blocks, int64_t threads);
double time_chase(torch::Tensor next, int64_t start, int64_t hops);
void run_stream3(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t kmax, int64_t policy, int64_t math, double out_scale, torch::Tensor prof);
double time_stream3(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax, int64_t policy, int64_t math);
void run_stream4(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t kmax, int64_t depth, int64_t rr, double out_scale, torch::Tensor prof);
double time_stream4(torch::Tensor descs, torch::Tensor stages, int64_t M, int64_t iters, int64_t kmax, int64_t depth, int64_t rr);
"""
    log(f"   building CUDA kernels for sm_{major}{minor} (first run takes ~1 min)...")
    return load_inline(
        name="cirrus_bench",
        cpp_sources=decls,
        cuda_sources=src,
        functions=["time_read", "time_coop_barrier", "time_flag_barrier", "nvfp4_gemv", "nvfp4_mma_gemv",
                   "run_stream", "time_stream", "run_stream2", "time_stream2", "time_read_rot", "time_chase",
                   "run_stream3", "time_stream3", "run_stream4", "time_stream4"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


@section("dram_bandwidth")
def dram_bandwidth(ext, sms: int) -> dict:
    buf = torch.empty(2 * 1024 * MB, dtype=torch.uint8, device="cuda")
    buf.random_(0, 255)
    best, rows = 0.0, []
    for bpsm in (2, 4, 8, 16):
        for threads in (256, 512, 1024):
            ms = min(ext.time_read(buf, 1, sms * bpsm, threads) for _ in range(3))
            gbps = buf.numel() / (ms * 1e-3) / 1e9
            rows.append({"blocks_per_sm": bpsm, "threads": threads, "gbps": round(gbps, 1)})
            best = max(best, gbps)
    # torch baselines
    t = buf.view(torch.float16)
    sum_ms = cuda_time_ms(lambda: t.sum(dtype=torch.float32), iters=5)
    dst = torch.empty_like(buf)
    copy_ms = cuda_time_ms(lambda: dst.copy_(buf), iters=5)
    out = {
        "best_read_gbps": round(best, 1),
        "efficiency_vs_273": round(best / 273.0, 3),
        "torch_sum_gbps": round(buf.numel() / (sum_ms * 1e-3) / 1e9, 1),
        "torch_copy_gbps_rw": round(2 * buf.numel() / (copy_ms * 1e-3) / 1e9, 1),
        "configs": rows,
    }
    log(f"   best streaming read: {out['best_read_gbps']} GB/s ({out['efficiency_vs_273']:.0%} of 273)")
    log(f"   torch sum: {out['torch_sum_gbps']} GB/s, torch copy (r+w): {out['torch_copy_gbps_rw']} GB/s")
    del buf, dst
    return out


@section("l2_sweep")
def l2_sweep(ext, sms: int) -> dict:
    # Random data: the first run used all-ones buffers and saw L2-like speed up
    # to 64 MB (> the 24 MB L2), so compare against constant data to rule out
    # hardware compression or another cache level.
    rows = []
    for size_mb in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 256):
        buf = torch.empty(size_mb * MB, dtype=torch.uint8, device="cuda").random_(0, 256)
        ones = torch.ones(size_mb * MB, dtype=torch.uint8, device="cuda")
        iters = max(1, (4096 // size_mb))  # ~4 GB of traffic per point
        ms = min(ext.time_read(buf, iters, sms * 8, 512) for _ in range(3))
        ms1 = min(ext.time_read(ones, iters, sms * 8, 512) for _ in range(3))
        gbps = size_mb * MB * iters / (ms * 1e-3) / 1e9
        gbps1 = size_mb * MB * iters / (ms1 * 1e-3) / 1e9
        rows.append({"mb": size_mb, "gbps": round(gbps, 1), "gbps_constant_data": round(gbps1, 1)})
        log(f"   {size_mb:4d} MB working set: random {gbps:8.1f} GB/s | constant {gbps1:8.1f} GB/s")
        del buf, ones
    dram = rows[-1]["gbps"]
    in_l2 = [r for r in rows if r["gbps"] > 1.5 * dram]
    return {
        "points": rows,
        "l2_plateau_gbps": round(statistics.median(r["gbps"] for r in in_l2), 1) if in_l2 else None,
        "largest_l2_resident_mb": max((r["mb"] for r in in_l2), default=None),
    }


@section("grid_barrier")
def grid_barrier(ext, sms: int) -> dict:
    n = 2000
    out = {}
    for name, fn in (("coop_grid_sync", ext.time_coop_barrier), ("atomic_flag", ext.time_flag_barrier)):
        for bpsm in (1, 2):
            base = min(fn(0, sms * bpsm, 256) for _ in range(5))
            full = min(fn(n, sms * bpsm, 256) for _ in range(5))
            us = (full - base) * 1e3 / n
            out[f"{name}_{bpsm}bpsm_us"] = round(us, 3)
            log(f"   {name} ({sms * bpsm} blocks): {us:.3f} us/barrier")
    out["best_us"] = min(v for k, v in out.items() if k.endswith("_us"))
    return out


@section("launch_overhead")
def launch_overhead() -> dict:
    x = torch.zeros(1, device="cuda")
    n = 200

    def eager():
        for _ in range(n):
            x.add_(1)

    eager_us = cuda_time_ms(eager, iters=10) * 1e3 / n
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        eager()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        eager()
    graph_us = cuda_time_ms(g.replay, iters=20) * 1e3 / n
    log(f"   eager launch: {eager_us:.2f} us/kernel, CUDA graph node: {graph_us:.2f} us/kernel")
    return {"eager_us_per_kernel": round(eager_us, 3), "graph_us_per_node": round(graph_us, 3)}


@section("host_roundtrip")
def host_roundtrip() -> dict:
    """GPU produces a token id -> CPU reads it -> CPU sends a draft back.
    Measured from Python, so it is an upper bound for the C++ engine."""
    dev = torch.zeros(16, dtype=torch.int32, device="cuda")
    host = torch.zeros(16, dtype=torch.int32, pin_memory=True)
    n = 2000
    for _ in range(50):
        dev.add_(1)
        host.copy_(dev, non_blocking=True)
        torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        dev.add_(1)
        host.copy_(dev, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        _ = int(host[0])
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / n
    log(f"   GPU->CPU->GPU per step (Python): {us:.1f} us")
    return {"roundtrip_us_python": round(us, 2)}


def make_nvfp4(n: int, k: int, gen: torch.Generator):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    scales = (torch.rand(n, k // 16, device="cuda", generator=gen) * 0.9 + 0.1).to(torch.float8_e4m3fn)
    return w, scales.view(torch.uint8)


def dequant_ref(w: torch.Tensor, s: torch.Tensor, gscale: float) -> torch.Tensor:
    mag = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    lut = torch.cat((mag, -mag))
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    vals = torch.stack((lut[lo], lut[hi]), dim=-1).flatten(-2)  # [N, K]
    sc = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sc * gscale


@section("nvfp4_gemv_correctness")
def nvfp4_correctness(ext) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(0)
    worst = 0.0
    for m in (1, 4, 13):
        for n, k in ((768, 768), (4096, 768), (768, 2048)):
            w, s = make_nvfp4(n, k, gen)
            x = torch.randn(m, k, device="cuda", generator=gen).half()
            y = ext.nvfp4_gemv(w, s, x, 0.5)
            ref = x.float() @ dequant_ref(w, s, 0.5).T
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, err)
    ok = worst < 1e-3
    log(f"   max relative error vs torch reference: {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"nvfp4_gemv mismatch: {worst}")
    return {"max_rel_err": worst}


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


@section("decode_weight_stream")
def decode_weight_stream(ext) -> dict:
    """Replay cirrus-s-1a's full decode-step GEMV sequence (random weights),
    one kernel per matrix, captured in a CUDA graph."""
    cfg = get_config("cirrus-s-1a")
    mats = decode_weight_matrices(cfg)
    gen = torch.Generator(device="cuda").manual_seed(1)
    out = {"n_matrices": len(mats)}

    # NVFP4, custom kernel
    q = [make_nvfp4(n, k, gen) for _, n, k in mats]
    fp4_bytes = sum(w.numel() + s.numel() for w, s in q)
    out["nvfp4_mb"] = round(fp4_bytes / 1e6, 1)
    for m in (1, 2, 4, 8, 13):
        xs = {k: torch.randn(m, k, device="cuda").half() for k in {k for _, _, k in mats}}
        g = capture(lambda: [ext.nvfp4_gemv(w, s, xs[w.shape[1] * 2], 1.0) for w, s in q])
        ms = cuda_time_ms(g.replay, iters=50)
        gbps = fp4_bytes / (ms * 1e-3) / 1e9
        out[f"nvfp4_M{m}_us"] = round(ms * 1e3, 1)
        out[f"nvfp4_M{m}_gbps"] = round(gbps, 1)
        log(f"   NVFP4 stream, M={m:2d}: {ms * 1e3:7.1f} us/step  ({gbps:6.1f} GB/s, {out['nvfp4_mb']} MB)")
    del q

    # BF16 via cuBLAS, the "plain PyTorch" baseline
    W = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _, n, k in mats]
    bf16_bytes = sum(w.numel() * 2 for w in W)
    for m in (1, 8):
        xs = {k: torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for k in {k for _, _, k in mats}}
        g = capture(lambda: [torch.nn.functional.linear(xs[w.shape[1]], w) for w in W])
        ms = cuda_time_ms(g.replay, iters=30)
        gbps = bf16_bytes / (ms * 1e-3) / 1e9
        out[f"bf16_M{m}_us"] = round(ms * 1e3, 1)
        out[f"bf16_M{m}_gbps"] = round(gbps, 1)
        log(f"   BF16 cuBLAS,  M={m:2d}: {ms * 1e3:7.1f} us/step  ({gbps:6.1f} GB/s, {bf16_bytes / 1e6:.0f} MB)")
    return out


# ------------------------------------------------- v1: tensor-core NVFP4
def pack_nvfp4_mma(codes: torch.Tensor, scales: torch.Tensor):
    """Standard NVFP4 ([N, K] E2M1 codes 0..15, [N, K/16] E4M3 scale bytes)
    -> the fragment-major layout read by mma_task in bench_kernels.cu."""
    N, K = codes.shape
    Np = (N + 15) // 16 * 16
    if Np != N:
        codes = torch.cat((codes, codes.new_zeros(Np - N, K)))
        scales = torch.cat((scales, scales.new_zeros(Np - N, K // 16)))
    NT, KT = Np // 16, K // 64
    C = codes.view(NT, 16, KT, 4, 16).long()  # [nt, row, kt, k16 block, col]
    lanes = []
    for lane in range(32):
        g, c = lane >> 2, lane & 3
        k0, k8 = 2 * c, 2 * c + 8
        nib = torch.stack(
            [C[:, g, :, :, k0], C[:, g, :, :, k0 + 1], C[:, g + 8, :, :, k0], C[:, g + 8, :, :, k0 + 1],
             C[:, g, :, :, k8], C[:, g, :, :, k8 + 1], C[:, g + 8, :, :, k8], C[:, g + 8, :, :, k8 + 1]],
            dim=-1,
        )  # [nt, kt, block j, 8 nibbles]
        lanes.append((nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8))  # [nt, kt, 4, 4]
    W = torch.stack(lanes, dim=2).reshape(-1)  # [nt, kt, lane, 16 bytes]
    S = scales.view(NT, 16, KT, 4).permute(0, 2, 1, 3).contiguous().reshape(-1)  # [nt, kt, row, block]
    return W.contiguous(), S


def random_nvfp4(n: int, k: int, gen: torch.Generator):
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device="cuda", generator=gen)
    scales = (torch.rand(n, k // 16, device="cuda", generator=gen) * 0.9 + 0.1).to(torch.float8_e4m3fn)
    return codes, scales.view(torch.uint8)


def dequant_codes(codes: torch.Tensor, scales: torch.Tensor, gscale: float = 1.0) -> torch.Tensor:
    mag = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=codes.device)
    lut = torch.cat((mag, -mag))
    sc = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return lut[codes.long()] * sc * gscale


def build_stream(ext, stages, M: int, gen, keep_ref: bool = False):
    """Allocate packed weights + I/O for a list of stages; return descriptors."""
    rows, firsts, mats, refs = [], [0], [], []
    kmax = max(k for st in stages for _, _, k in st)
    xs = {k: torch.randn(M, k, device="cuda", generator=gen).half() for k in {k for st in stages for _, _, k in st}}
    y_total = sum(M * n for st in stages for _, n, _ in st)
    Y = torch.zeros(y_total, device="cuda")
    off = 0
    for s, st in enumerate(stages):
        task_off = 0
        for name, n, k in st:
            codes, scales = random_nvfp4(n, k, gen)
            W, S = pack_nvfp4_mma(codes, scales)
            y = Y[off : off + M * n]
            off += M * n
            rows.append([W.data_ptr(), S.data_ptr(), xs[k].data_ptr(), y.data_ptr(), n | (k << 32), s | (task_off << 32)])
            task_off += ((n + 15) // 16) * (k // 128)
            mats.append((W, S))
            if keep_ref:
                refs.append((y, xs[k].float() @ dequant_codes(codes, scales).T))
        firsts.append(len(rows))
    descs = torch.tensor(rows, dtype=torch.int64, device="cuda")
    sf = torch.tensor(firsts, dtype=torch.int32, device="cuda")
    return {"descs": descs, "stage_first": sf, "n_stages": len(stages), "kmax": kmax, "Y": Y, "keep": (mats, xs),
            "refs": refs, "bytes": sum(W.numel() + S.numel() for W, S in mats)}


@section("nvfp4_mma_correctness")
def nvfp4_mma_correctness(ext) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(2)
    worst = 0.0
    for m in (1, 4, 8, 13):
        for n, k in ((768, 768), (12, 768), (4096, 768), (768, 2048), (768, 1536)):
            codes, scales = random_nvfp4(n, k, gen)
            W, S = pack_nvfp4_mma(codes, scales)
            x = torch.randn(m, k, device="cuda", generator=gen).half()
            y = torch.zeros(m, n, device="cuda")
            ext.nvfp4_mma_gemv(W, S, x, n, 0.5, y)
            ref = x.float() @ dequant_codes(codes, scales, 0.5).T
            worst = max(worst, ((y - ref).abs().max() / ref.abs().max()).item())
    # persistent kernel on a small multi-stage program
    stages = [[("a", 768, 768), ("b", 12, 768)], [("c", 768, 2048)], [("d", 4096, 768)]]
    worst_stream = 0.0
    for m in (1, 13):
        st = build_stream(ext, stages, m, gen, keep_ref=True)
        ext.run_stream(st["descs"], st["stage_first"], st["n_stages"], m, st["kmax"])
        torch.cuda.synchronize()
        for y, ref in st["refs"]:
            worst_stream = max(worst_stream, ((y.view_as(ref) - ref).abs().max() / ref.abs().max()).item())
    ok = worst < 1e-3 and worst_stream < 1e-3
    log(f"   mma GEMV max rel err: {worst:.2e}; persistent stream: {worst_stream:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"mma kernels mismatch: gemv {worst}, stream {worst_stream}")
    return {"gemv_max_rel_err": worst, "stream_max_rel_err": worst_stream}


@section("decode_stream_v1")
def decode_stream_v1(ext) -> dict:
    """cirrus-s-1a's full decode-step weight stream with the tensor-core kernel:
    (a) one kernel per matrix in a CUDA graph, (b) ONE persistent cooperative
    kernel with a grid barrier per fused stage (the megakernel's skeleton)."""
    from cirrus.perf.shapes import decode_stages

    cfg = get_config("cirrus-s-1a")
    stages = decode_stages(cfg)
    gen = torch.Generator(device="cuda").manual_seed(3)
    out = {"n_stages": len(stages)}
    for m in (1, 2, 4, 8, 13):
        st = build_stream(ext, stages, m, gen)
        mb = st["bytes"] / 1e6
        iters = 50
        ms = min(ext.time_stream(st["descs"], st["stage_first"], st["n_stages"], m, iters, st["kmax"]) for _ in range(3))
        us = ms * 1e3 / iters
        out[f"persistent_M{m}_us"] = round(us, 1)
        out[f"persistent_M{m}_gbps"] = round(st["bytes"] / (us * 1e-6) / 1e9, 1)
        # same work, one kernel per matrix, CUDA graph
        mats, xs = st["keep"]
        flat = [(W, S, n, k) for (W, S), (_, n, k) in zip(mats, [x for s in stages for x in s])]
        Y = st["Y"]

        def per_matrix():
            Y.zero_()
            off = 0
            for W, S, n, k in flat:
                ext.nvfp4_mma_gemv(W, S, xs[k], n, 1.0, Y[off : off + m * n].view(m, n))
                off += m * n

        g = capture(per_matrix)
        gms = cuda_time_ms(g.replay, iters=30)
        out[f"graph_M{m}_us"] = round(gms * 1e3, 1)
        out[f"graph_M{m}_gbps"] = round(st["bytes"] / (gms * 1e-3) / 1e9, 1)
        log(f"   M={m:2d}: persistent {us:7.1f} us ({out[f'persistent_M{m}_gbps']:6.1f} GB/s) | "
            f"graph {gms * 1e3:7.1f} us ({out[f'graph_M{m}_gbps']:6.1f} GB/s) | {mb:.1f} MB")
        del st, g
    out["weight_mb"] = round(mb, 1)
    return out


# ------------------------------------------- v2: pipelined persistent stream
POLICIES = {0: "evict_first(.cs)", 1: "l2(.cg)", 2: "readonly(.nc)"}


def make_tasks(stages, nwarps: int):
    """Split-K sized per stage so the stage has ~nwarps tasks (tile-major,
    so one tile's K-splits land on neighbouring warps)."""
    rows, firsts, mi = [], [0], 0
    for st in stages:
        total_chunks = sum(((n + 15) // 16) * (k // 128) for _, n, k in st)
        nkc = max(1, -(-total_chunks // nwarps))
        for _, n, k in st:
            kchunks = k // 128
            step = min(nkc, kchunks)
            for nt in range((n + 15) // 16):
                for kc0 in range(0, kchunks, step):
                    rows.append([mi, nt, kc0, min(step, kchunks - kc0)])
            mi += 1
        firsts.append(len(rows))
    return (torch.tensor(rows, dtype=torch.int32, device="cuda"),
            torch.tensor(firsts, dtype=torch.int32, device="cuda"))


def nwarps_total() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count * 16  # 512-thread blocks


@section("stream_v2_correctness")
def stream_v2_correctness(ext) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(4)
    stages = [[("a", 768, 768), ("b", 12, 768)], [("c", 768, 2048)], [("d", 4096, 768)], [("e", 64, 1536)]]
    worst = 0.0
    for m in (1, 5, 13):
        for policy in POLICIES:
            st = build_stream(ext, stages, m, gen, keep_ref=True)
            tasks, stf = make_tasks(stages, nwarps_total())
            ext.run_stream2(st["descs"], st["stage_first"], tasks, stf, st["n_stages"], m, st["kmax"], policy)
            torch.cuda.synchronize()
            for y, ref in st["refs"]:
                worst = max(worst, ((y.view_as(ref) - ref).abs().max() / ref.abs().max()).item())
    ok = worst < 1e-3
    log(f"   pipelined stream max rel err (3 policies, M=1/5/13): {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"stream2 mismatch: {worst}")
    return {"max_rel_err": worst}


@section("decode_stream_v2")
def decode_stream_v2(ext) -> dict:
    from cirrus.perf.shapes import decode_stages

    cfg = get_config("cirrus-s-1a")
    stages = decode_stages(cfg)
    gen = torch.Generator(device="cuda").manual_seed(5)
    tasks, stf = make_tasks(stages, nwarps_total())
    out = {"n_stages": len(stages), "n_tasks": int(tasks.shape[0])}
    iters = 50
    for m in (1, 2, 4, 8, 13):
        st = build_stream(ext, stages, m, gen)
        line = [f"   M={m:2d}:"]
        for policy, pname in POLICIES.items():
            ms = min(ext.time_stream2(st["descs"], st["stage_first"], tasks, stf, st["n_stages"], m, iters,
                                      st["kmax"], policy) for _ in range(3))
            us = ms * 1e3 / iters
            out[f"p{policy}_M{m}_us"] = round(us, 1)
            out[f"p{policy}_M{m}_gbps"] = round(st["bytes"] / (us * 1e-6) / 1e9, 1)
            line.append(f"{pname} {us:6.1f} us ({out[f'p{policy}_M{m}_gbps']:6.1f} GB/s)")
        log(" | ".join(line))
        out["weight_mb"] = round(st["bytes"] / 1e6, 1)
        del st
    for m in (1, 2, 4, 8, 13):  # best policy per M, in the v1 key format for the estimator
        best = min(out[f"p{p}_M{m}_us"] for p in POLICIES)
        out[f"persistent_M{m}_us"] = best
    return out


@section("cache_resident_stream")
def cache_resident_stream(ext) -> dict:
    """If GB10 really caches > 24 MB (L2 sweep, run 2), a decode step whose
    weights fit could run far above DRAM speed. Stream the first stages of
    the model up to a byte budget, back to back, and compare cache hints."""
    from cirrus.perf.shapes import decode_stages

    stages_all = decode_stages(get_config("cirrus-s-1a"))
    gen = torch.Generator(device="cuda").manual_seed(6)
    out = {}
    for budget_mb in (16, 32, 48, 64, 80):
        stages, total = [], 0
        for s in stages_all:
            b = sum(((n + 15) // 16 * 16) * k * 9 // 16 for _, n, k in s)  # 4.5 bits/weight
            if total + b > budget_mb * 1e6:
                break
            stages.append(s)
            total += b
        st = build_stream(ext, stages, 1, gen)
        tasks, stf = make_tasks(stages, nwarps_total())
        line = [f"   {budget_mb:3d} MB budget ({st['bytes'] / 1e6:5.1f} MB, {len(stages)} stages):"]
        for policy, pname in POLICIES.items():
            ms = min(ext.time_stream2(st["descs"], st["stage_first"], tasks, stf, st["n_stages"], 1, 50,
                                      st["kmax"], policy) for _ in range(3))
            gbps = st["bytes"] * 50 / (ms * 1e-3) / 1e9
            out[f"{budget_mb}MB_p{policy}_gbps"] = round(gbps, 1)
            line.append(f"{pname} {gbps:7.1f} GB/s")
        log(" | ".join(line))
        del st
    return out


@section("cache_probe")
def cache_probe(ext, sms: int) -> dict:
    out = {"rotated_read": [], "chase": []}
    for size_mb in (16, 24, 32, 40, 48, 64, 80, 96, 128, 256):
        buf = torch.empty(size_mb * MB, dtype=torch.uint8, device="cuda").random_(0, 256)
        iters = max(2, 4096 // size_mb)
        ms = min(ext.time_read_rot(buf, iters, sms * 8, 512) for _ in range(3))
        gbps = size_mb * MB * iters / (ms * 1e-3) / 1e9
        out["rotated_read"].append({"mb": size_mb, "gbps": round(gbps, 1)})
        del buf
    log("   rotated-offset read: " + ", ".join(f"{r['mb']}MB {r['gbps']:.0f}" for r in out["rotated_read"]))
    hops = 100_000
    for size_mb in (0.25, 1, 4, 16, 24, 32, 40, 48, 64, 96, 128, 256):
        lines = int(size_mb * MB) // 128
        perm = torch.randperm(lines, device="cuda", dtype=torch.int64)
        nxt = torch.zeros(lines * 32, dtype=torch.int32, device="cuda")
        nxt[perm * 32] = (torch.roll(perm, -1) * 32).to(torch.int32)
        ms = min(ext.time_chase(nxt, int(perm[0]) * 32, hops) for _ in range(2))
        ns = ms * 1e6 / hops
        out["chase"].append({"mb": size_mb, "ns": round(ns, 1)})
        del nxt, perm
    log("   dependent-load latency (ns): " + ", ".join(f"{r['mb']}MB {r['ns']:.0f}" for r in out["chase"]))
    return out


# ----------------------------------------- v3: latency-lean persistent stream
def pack_nvfp4_v3(codes: torch.Tensor, scales: torch.Tensor):
    """NVFP4 -> layout v3 (see the v3 comment block in bench_kernels.cu)."""
    N, K = codes.shape
    Np = (N + 15) // 16 * 16
    if Np != N:
        codes = torch.cat((codes, codes.new_zeros(Np - N, K)))
        scales = torch.cat((scales, scales.new_zeros(Np - N, K // 16)))
    NT, KT = Np // 16, K // 64
    C = codes.view(NT, 16, KT, 4, 16).long()
    lanes = []
    for lane in range(32):
        g, c = lane >> 2, lane & 3
        k0, k8 = 2 * c, 2 * c + 8
        nib = torch.stack(
            [C[:, g, :, :, k0], C[:, g + 8, :, :, k0], C[:, g, :, :, k8], C[:, g + 8, :, :, k8],
             C[:, g, :, :, k0 + 1], C[:, g + 8, :, :, k0 + 1], C[:, g, :, :, k8 + 1], C[:, g + 8, :, :, k8 + 1]],
            dim=-1,
        )
        lanes.append((nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8))
    W = torch.stack(lanes, dim=2).reshape(-1)
    S = scales.view(NT, 2, 8, KT, 4).permute(0, 3, 2, 4, 1).contiguous().reshape(-1)  # [nt, kt, g, j, row half]
    return W.contiguous(), S


def build_stream3(stages, M: int, gen, nwarps: int, keep_ref: bool = False):
    kmax = max(k for st in stages for _, _, k in st)
    xs = {k: torch.randn(M, k, device="cuda", generator=gen).half() for k in {k for st in stages for _, _, k in st}}
    Y = torch.zeros(sum(M * n for st in stages for _, n, _ in st), device="cuda")
    rows, srows, keep, refs, off = [], [], [], [], 0
    for st in stages:
        m0 = len(rows)
        total_chunks = sum(((n + 15) // 16) * (k // 128) for _, n, k in st)
        nkc = max(1, -(-total_chunks // nwarps))
        task_off = 0
        for _, n, k in st:
            codes, scales = random_nvfp4(n, k, gen)
            W, S = pack_nvfp4_v3(codes, scales)
            y = Y[off : off + M * n]
            off += M * n
            kchunks = k // 128
            splits = -(-kchunks // nkc)
            rows.append([W.data_ptr(), S.data_ptr(), xs[k].data_ptr(), y.data_ptr(), n | (k << 32),
                         task_off | (splits << 32), kchunks])
            task_off += ((n + 15) // 16) * splits
            keep.append((W, S))
            if keep_ref:
                refs.append((y, xs[k].float() @ dequant_codes(codes, scales).T))
        srows.append([m0, len(rows), nkc, task_off])
    return {"descs": torch.tensor(rows, dtype=torch.int64, device="cuda"),
            "stages": torch.tensor(srows, dtype=torch.int32, device="cuda"),
            "kmax": kmax, "Y": Y, "keep": (keep, xs), "refs": refs,
            "bytes": sum(W.numel() + S.numel() for W, S in keep)}


@section("stream_v3_correctness")
def stream_v3_correctness(ext) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(7)
    stages = [[("a", 768, 768), ("b", 12, 768)], [("c", 768, 2048)], [("d", 4096, 768)], [("e", 64, 1536)]]
    worst = 0.0
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    for m in (1, 5, 8, 13):
        for policy in (0, 1):
            st = build_stream3(stages, m, gen, nwarps_total(), keep_ref=True)
            # products carry v*s*2^-7 (fp4 bit trick 2^-14 x scale 2^7): undo with 128
            ext.run_stream3(st["descs"], st["stages"], m, st["kmax"], policy, 1, 128.0, empty)
            torch.cuda.synchronize()
            for y, ref in st["refs"]:
                worst = max(worst, ((y.view_as(ref) - ref).abs().max() / ref.abs().max()).item())
    ok = worst < 1e-3
    log(f"   v3 stream max rel err (2 policies, M=1/5/8/13): {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"stream3 mismatch: {worst}")
    return {"max_rel_err": worst}


def profile_stream3(ext, st, M: int, stages, policy: int, runner=None) -> dict:
    """Per-stage breakdown from in-kernel clock64 stamps (first iteration).
    ``runner(prof)`` launches the kernel to profile (default: v3)."""
    nb = torch.cuda.get_device_properties(0).multi_processor_count
    ns = len(stages)
    prof = torch.zeros(4 + ns * nb * 3, dtype=torch.int64, device="cuda")
    st["Y"].zero_()
    if runner is None:
        ext.run_stream3(st["descs"], st["stages"], M, st["kmax"], policy, 1, 1.0, prof)
    else:
        runner(prof)
    torch.cuda.synchronize()
    p = prof.cpu()
    cyc_per_ns = (p[2] - p[0]).item() / max(1, (p[3] - p[1]).item())
    t = p[4:].view(ns, nb, 3).double() / cyc_per_ns / 1e3  # us, per-SM clocks
    x_load = (t[:, :, 1] - t[:, :, 0])            # activation load + block sync
    compute = (t[:, :, 2] - t[:, :, 1])           # this block's chunks
    stage_span = torch.cat((t[1:, 0, 0] - t[:-1, 0, 0], ((p[2] - p[0]).double() / cyc_per_ns / 1e3 - (t[-1, 0, 0] - t[0, 0, 0])).view(1)))
    busy = (t[:, :, 2] - t[:, :, 0]).max(dim=1).values
    rows = []
    for i, s in enumerate(stages):
        b = sum(((n + 15) // 16 * 16) * k * 9 // 16 for _, n, k in s)
        rows.append({"stage": "+".join(n for n, _, _ in s), "mb": round(b / 1e6, 3),
                     "span_us": round(stage_span[i].item(), 2), "xload_us": round(x_load[i].median().item(), 2),
                     "compute_max_us": round(compute[i].max().item(), 2),
                     "compute_med_us": round(compute[i].median().item(), 2),
                     "wait_us": round((stage_span[i] - busy[i]).item(), 2)})
    tot = lambda k: round(sum(r[k] for r in rows), 1)  # noqa: E731
    summary = {"sm_ghz": round(cyc_per_ns, 3), "total_us": tot("span_us"), "xload_us": tot("xload_us"),
               "compute_max_us": tot("compute_max_us"), "compute_med_us": tot("compute_med_us"),
               "barrier_and_imbalance_us": tot("wait_us")}
    slow = sorted(rows, key=lambda r: -r["span_us"])[:8]
    return {"summary": summary, "slowest": slow, "stages": rows}


@section("decode_stream_v3")
def decode_stream_v3(ext) -> dict:
    from cirrus.perf.shapes import decode_stages

    cfg = get_config("cirrus-s-1a")
    stages = decode_stages(cfg)
    gen = torch.Generator(device="cuda").manual_seed(8)
    out = {"n_stages": len(stages)}
    iters = 50
    for m in (1, 2, 4, 8, 13):
        st = build_stream3(stages, m, gen, nwarps_total())
        out["weight_mb"] = round(st["bytes"] / 1e6, 1)
        line = [f"   M={m:2d}:"]
        for policy, math in ((0, 1), (1, 1), (0, 0)):
            ms = min(ext.time_stream3(st["descs"], st["stages"], m, iters, st["kmax"], policy, math) for _ in range(3))
            us = ms * 1e3 / iters
            key = f"p{policy}{'' if math else '_nomath'}_M{m}"
            out[f"{key}_us"] = round(us, 1)
            out[f"{key}_gbps"] = round(st["bytes"] / (us * 1e-6) / 1e9, 1)
            label = {(0, 1): ".cs", (1, 1): ".cg", (0, 0): ".cs no-math"}[(policy, math)]
            line.append(f"{label} {us:6.1f} us ({out[f'{key}_gbps']:6.1f} GB/s)")
        log(" | ".join(line))
        if m in (1, 4):
            prof = profile_stream3(ext, st, m, stages, 0)
            out[f"profile_M{m}"] = prof
            s = prof["summary"]
            log(f"      profile M={m} (SM clock {s['sm_ghz']} GHz): stages {s['total_us']} us = "
                f"x-load {s['xload_us']} + compute(max) {s['compute_max_us']} [median {s['compute_med_us']}] "
                f"+ barrier/imbalance {s['barrier_and_imbalance_us']}")
            for r in prof["slowest"][:5]:
                log(f"        slow: {r['stage'][:40]:40s} {r['mb']:6.2f} MB span {r['span_us']:6.2f} us "
                    f"(x {r['xload_us']}, compute max {r['compute_max_us']}, wait {r['wait_us']})")
        del st
    for m in (1, 2, 4, 8, 13):
        out[f"persistent_M{m}_us"] = min(out[f"p0_M{m}_us"], out[f"p1_M{m}_us"])
    return out


# ------------------------------------- v4: placement + prefetch-depth sweep
def v4_depths(m: int):
    return (1, 2, 3) if m <= 8 else (1, 2)


@section("stream_v4_correctness")
def stream_v4_correctness(ext) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(9)
    stages = [[("a", 768, 768), ("b", 12, 768)], [("c", 768, 2048)], [("d", 4096, 768)], [("e", 64, 1536)]]
    worst = 0.0
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    for m in (1, 5, 13):
        st = build_stream3(stages, m, gen, nwarps_total(), keep_ref=True)
        for depth in v4_depths(m):
            for rr in (0, 1):
                st["Y"].zero_()
                ext.run_stream4(st["descs"], st["stages"], m, st["kmax"], depth, rr, 128.0, empty)
                torch.cuda.synchronize()
                for y, ref in st["refs"]:
                    worst = max(worst, ((y.view_as(ref) - ref).abs().max() / ref.abs().max()).item())
    ok = worst < 1e-3
    log(f"   v4 stream max rel err (depth 1-3, both placements, M=1/5/13): {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"stream4 mismatch: {worst}")
    return {"max_rel_err": worst}


@section("decode_stream_v4")
def decode_stream_v4(ext) -> dict:
    from cirrus.perf.shapes import decode_stages

    cfg = get_config("cirrus-s-1a")
    stages = decode_stages(cfg)
    gen = torch.Generator(device="cuda").manual_seed(10)
    out = {"n_stages": len(stages)}
    iters = 50
    for m in (1, 2, 4, 8, 13):
        st = build_stream3(stages, m, gen, nwarps_total())
        out["weight_mb"] = round(st["bytes"] / 1e6, 1)
        line, best = [f"   M={m:2d}:"], None
        for rr in (0, 1):
            for depth in v4_depths(m):
                ms = min(ext.time_stream4(st["descs"], st["stages"], m, iters, st["kmax"], depth, rr) for _ in range(3))
                us = ms * 1e3 / iters
                key = f"rr{rr}_d{depth}_M{m}"
                out[f"{key}_us"] = round(us, 1)
                out[f"{key}_gbps"] = round(st["bytes"] / (us * 1e-6) / 1e9, 1)
                line.append(f"{'rr' if rr else 'wm'} d{depth} {us:6.1f}")
                if best is None or us < best[0]:
                    best = (us, rr, depth)
        out[f"persistent_M{m}_us"] = round(best[0], 1)
        out[f"best_M{m}"] = {"us": round(best[0], 1), "rr": best[1], "depth": best[2],
                             "gbps": round(st["bytes"] / (best[0] * 1e-6) / 1e9, 1)}
        log(" | ".join(line) + f"   -> best {'rr' if best[1] else 'wm'} d{best[2]}: "
            f"{best[0]:.1f} us ({out[f'best_M{m}']['gbps']} GB/s)")
        if m in (1, 4):
            _, rr, depth = best
            prof = profile_stream3(ext, st, m, stages, 0, runner=lambda pr: ext.run_stream4(
                st["descs"], st["stages"], m, st["kmax"], depth, rr, 1.0, pr))
            out[f"profile_M{m}"] = prof
            s = prof["summary"]
            log(f"      profile M={m} ({'rr' if rr else 'wm'} d{depth}): stages {s['total_us']} us = x-load "
                f"{s['xload_us']} + compute(max) {s['compute_max_us']} [median {s['compute_med_us']}] "
                f"+ barrier/imbalance {s['barrier_and_imbalance_us']}")
            for r in prof["slowest"][:5]:
                log(f"        slow: {r['stage'][:40]:40s} {r['mb']:6.2f} MB span {r['span_us']:6.2f} us "
                    f"(x {r['xload_us']}, compute max {r['compute_max_us']}, med {r['compute_med_us']}, wait {r['wait_us']})")
        del st
    return out


@section("prefill_gemm")
def prefill_gemm() -> dict:
    T = 16384
    out = {}
    for n, k in ((2048 * 2, 768), (768, 2048), (2304, 768)):
        a = torch.randn(T, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ms = cuda_time_ms(lambda: a @ b.T, iters=20)
        tf = 2 * T * n * k / (ms * 1e-3) / 1e12
        out[f"bf16_{n}x{k}_tflops"] = round(tf, 1)
        log(f"   BF16 [{T}x{k}] @ [{k}x{n}]: {tf:.1f} TFLOPS")
        try:
            a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
            one = torch.ones((), device="cuda")
            f = lambda: torch._scaled_mm(a8, b8.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
            ms8 = cuda_time_ms(f, iters=20)
            tf8 = 2 * T * n * k / (ms8 * 1e-3) / 1e12
            out[f"fp8_{n}x{k}_tflops"] = round(tf8, 1)
            log(f"   FP8  [{T}x{k}] @ [{k}x{n}]: {tf8:.1f} TFLOPS")
        except Exception as e:
            out["fp8_error"] = str(e)[:200]
    return out


@section("model_smoke_test")
def model_smoke_test() -> dict:
    """The reference model has only ever been checked by inspection; make
    sure it runs on the GPU and that its caches agree with a full forward."""
    import cirrus.model.layers as L
    from cirrus.model.model import CirrusModel

    torch.manual_seed(0)
    m = CirrusModel(get_config("cirrus-s-1a")).cuda().eval()
    ids = torch.randint(0, m.cfg.vocab_size, (1, 96), device="cuda")
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()  # noqa: E731
    fla_kernel, L._fla_chunk_gdr = L._fla_chunk_gdr, None  # exact reference path first
    try:
        with torch.no_grad():
            full = m(ids).logits
            o = m(ids[:, :64], use_cache=True)
            parts, cache = [o.logits], o.cache
            for t in range(64, 96):
                o = m(ids[:, t : t + 1], cache=cache)
                parts.append(o.logits)
                cache = o.cache
        err = rel(torch.cat(parts, 1), full)
    finally:
        L._fla_chunk_gdr = fla_kernel
    out = {"params": m.num_params(), "incremental_rel_err": err}
    log(f"   params: {m.num_params() / 1e6:.1f}M, incremental-vs-full max rel err: {err:.2e}")
    if fla_kernel is not None:
        try:
            with torch.no_grad():
                out["fla_vs_reference_rel_err"] = rel(m(ids).logits, full)
            log(f"   fla chunked GDN vs reference: {out['fla_vs_reference_rel_err']:.2e}")
        except Exception as e:
            out["fla_error"] = f"{type(e).__name__}: {e}"[:300]
            log(f"   fla kernel failed on this GPU: {out['fla_error']}")
    if err > 1e-3:
        raise AssertionError(f"incremental decode disagrees with full forward: {err}")
    return out


def measured_spec(env, dram, l2, bar, launch, host):
    hw = DGX_SPARK
    upd = {}
    if dram:
        upd["dram_efficiency"] = min(1.0, dram["best_read_gbps"] / hw.dram_bw_gbps)
    if env and env.get("l2_bytes"):
        upd["l2_bytes"] = int(env["l2_bytes"])
    if l2 and l2.get("l2_plateau_gbps"):
        upd["l2_bw_gbps"] = l2["l2_plateau_gbps"]
    if bar:
        upd["grid_barrier_us"] = bar["best_us"]
    if launch:
        upd["kernel_launch_us"] = launch["graph_us_per_node"]
    if host:
        upd["host_sync_us"] = host["roundtrip_us_python"]
    return replace(hw, name="DGX Spark (measured)", **upd), upd


def predictions(hw) -> dict:
    cfg = get_config("cirrus-s-1a")
    rows = {}
    for tau in (1.0, 1.65, 2.0, 2.5):
        for ctx in (4096, 32768, 65536):
            s = decode_step(cfg, hw, InferencePlan(tokens_per_step=tau, use_mtp=tau > 1), ctx)
            rows[f"tau{tau}_ctx{ctx}"] = round(s.tokens_per_second)
    return rows


def stream_based_estimate(hw, stream: dict) -> dict:
    """Decode tok/s where the weight stream (the dominant ~85% of bytes, plus
    its barriers) is the measured persistent-kernel time and only the small
    KV/state traffic still comes from the roofline model."""
    cfg = get_config("cirrus-s-1a")
    weight_keys = ("weights.body", "lm_head", "mtp.weights", "mtp.draft_head")
    out = {}
    for tau, m in ((1.0, 1), (1.65, 4), (2.0, 4), (2.5, 8), (3.0, 13)):
        key = f"persistent_M{m}_us"
        if key not in stream:
            continue
        for ctx in (4096, 32768, 65536):
            s = decode_step(cfg, hw, InferencePlan(tokens_per_step=tau, use_mtp=tau > 1, n_verify=m), ctx)
            other = s.total_dram_bytes - sum(s.dram_bytes.get(k, 0) for k in weight_keys)
            t_other = other / hw.eff_dram_bw + sum(s.l2_bytes.values()) / hw.eff_l2_bw
            # mixer stages (GDN recurrence, attention) add ~1 barrier per layer;
            # the 2nd MTP pass adds 4 stages; plus the host round trip
            barriers = cfg.n_layers + (cfg.mtp_passes - 1) * 4 if tau > 1 else cfg.n_layers
            t_fixed = (barriers * hw.grid_barrier_us + hw.host_sync_us) * 1e-6
            step = stream[key] * 1e-6 + t_other + t_fixed
            out[f"tau{tau}_verify{m}_ctx{ctx}"] = round(tau / step)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true", help="skip the reference-model smoke test")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these sections (plus environment), e.g. --only l2_sweep decode_stream_v1")
    ap.add_argument("--out", default=str(ROOT / "results" / "local"))  # gitignored
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to this torch build")

    t0 = time.time()
    env = environment()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ext = None
    try:
        ext = build_ext()
    except Exception as e:
        RESULTS["errors"]["build_ext"] = f"{type(e).__name__}: {e}"
        log(f"   kernel build FAILED ({e}); custom-kernel sections will be skipped")

    want = lambda name: args.only is None or name in args.only  # noqa: E731
    dram = dram_bandwidth(ext, sms) if ext and want("dram_bandwidth") else None
    l2 = l2_sweep(ext, sms) if ext and want("l2_sweep") else None
    bar = grid_barrier(ext, sms) if ext and want("grid_barrier") else None
    launch = launch_overhead() if want("launch_overhead") else None
    host = host_roundtrip() if want("host_roundtrip") else None
    if ext and want("decode_weight_stream") and nvfp4_correctness(ext) is not None:
        decode_weight_stream(ext)
    stream = None
    if ext and want("decode_stream_v1") and nvfp4_mma_correctness(ext) is not None:
        stream = decode_stream_v1(ext)
    if ext and want("decode_stream_v2") and stream_v2_correctness(ext) is not None:
        stream = decode_stream_v2(ext) or stream
    if ext and want("decode_stream_v3") and stream_v3_correctness(ext) is not None:
        stream = decode_stream_v3(ext) or stream
    if ext and want("decode_stream_v4") and stream_v4_correctness(ext) is not None:
        stream = decode_stream_v4(ext) or stream
    if ext and want("cache_resident_stream"):
        cache_resident_stream(ext)
    if ext and want("cache_probe"):
        cache_probe(ext, sms)
    if want("prefill_gemm"):
        prefill_gemm()
    if not args.skip_model and want("model_smoke_test"):
        model_smoke_test()

    hw, upd = measured_spec(env, dram, l2, bar, launch, host)
    RESULTS["measured_hardware"] = asdict(hw)
    RESULTS["predicted_tok_s_estimated_hw"] = predictions(DGX_SPARK)
    RESULTS["predicted_tok_s_measured_hw"] = predictions(hw)
    if stream:
        RESULTS["tok_s_from_measured_stream"] = stream_based_estimate(hw, stream)
    RESULTS["elapsed_s"] = round(time.time() - t0, 1)

    log("\n== prediction with MEASURED hardware (cirrus-s-1a, tok/s) ==")
    est, mea = RESULTS["predicted_tok_s_estimated_hw"], RESULTS["predicted_tok_s_measured_hw"]
    log(f"   {'case':24s} {'estimated':>10s} {'measured':>10s}")
    for k in mea:
        log(f"   {k:24s} {est[k]:>10,d} {mea[k]:>10,d}")
    log(f"   measured overrides: {upd}")
    if stream:
        log("\n== tok/s using the MEASURED persistent weight stream (+ roofline for KV/state) ==")
        for k, v in RESULTS["tok_s_from_measured_stream"].items():
            log(f"   {k:28s} {v:>8,d}")
    if RESULTS["errors"]:
        log(f"\n   sections with errors: {list(RESULTS['errors'])}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spark-{socket.gethostname()}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(RESULTS, indent=2, default=str))
    log(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
