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
"""
    log(f"   building CUDA kernels for sm_{major}{minor} (first run takes ~1 min)...")
    return load_inline(
        name="cirrus_bench",
        cpp_sources=decls,
        cuda_sources=src,
        functions=["time_read", "time_coop_barrier", "time_flag_barrier", "nvfp4_gemv"],
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
    rows = []
    for size_mb in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 128, 256):
        buf = torch.ones(size_mb * MB, dtype=torch.uint8, device="cuda")
        iters = max(1, (4096 // size_mb))  # ~4 GB of traffic per point
        ms = min(ext.time_read(buf, iters, sms * 8, 512) for _ in range(3))
        gbps = size_mb * MB * iters / (ms * 1e-3) / 1e9
        rows.append({"mb": size_mb, "gbps": round(gbps, 1)})
        log(f"   {size_mb:4d} MB working set: {gbps:8.1f} GB/s")
        del buf
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true", help="skip the reference-model smoke test")
    ap.add_argument("--out", default=str(ROOT / "results"))
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

    dram = dram_bandwidth(ext, sms) if ext else None
    l2 = l2_sweep(ext, sms) if ext else None
    bar = grid_barrier(ext, sms) if ext else None
    launch = launch_overhead()
    host = host_roundtrip()
    if ext and nvfp4_correctness(ext) is not None:
        decode_weight_stream(ext)
    prefill_gemm()
    if not args.skip_model:
        model_smoke_test()

    hw, upd = measured_spec(env, dram, l2, bar, launch, host)
    RESULTS["measured_hardware"] = asdict(hw)
    RESULTS["predicted_tok_s_estimated_hw"] = predictions(DGX_SPARK)
    RESULTS["predicted_tok_s_measured_hw"] = predictions(hw)
    RESULTS["elapsed_s"] = round(time.time() - t0, 1)

    log("\n== prediction with MEASURED hardware (cirrus-s-1a, tok/s) ==")
    est, mea = RESULTS["predicted_tok_s_estimated_hw"], RESULTS["predicted_tok_s_measured_hw"]
    log(f"   {'case':24s} {'estimated':>10s} {'measured':>10s}")
    for k in mea:
        log(f"   {k:24s} {est[k]:>10,d} {mea[k]:>10,d}")
    log(f"   measured overrides: {upd}")
    if RESULTS["errors"]:
        log(f"\n   sections with errors: {list(RESULTS['errors'])}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spark-{socket.gethostname()}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(RESULTS, indent=2, default=str))
    log(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
