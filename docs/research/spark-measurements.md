# DGX Spark measurements

Collected with `scripts/bench_spark.py` on the GPU of one DGX Spark ("grace"): DGX OS 7.2.3,
driver 580.178.04, CUDA 13.0, torch 2.13.0+cu130, GB10 at sm_121, 48 SMs. Raw data lives in
`results/`.

## Run 1 (2026-10-23): hardware characterization

| quantity | assumed | **measured** | notes |
|---|---|---|---|
| streaming DRAM read | 218 GB/s (80%) | **238.4 GB/s (87.3% of 273)** | flat across launch configs; `torch.sum` hits the same |
| copy (read + write) | | 219.1 GB/s | |
| L2 size | 24 MB | **24 MiB** (`l2CacheSize`) | |
| L2 bandwidth | 1,500 GB/s | **~2,040 GB/s** | plateau, 1-32 MB working sets |
| grid barrier, `grid.sync()` | 1.0 µs | **0.61 µs** | 48 or 96 blocks |
| grid barrier, atomic flag | | 1.03 µs | the cooperative barrier wins; use it |
| CUDA graph node | 2.5 µs | **0.94 µs** | eager launch 3.3 µs |
| GPU→CPU→GPU round trip | 15 µs | **12.5 µs** | from Python; C++ will be lower |
| prefill GEMM, BF16 | 125×0.45 TFLOPS | **65-93 TFLOPS** | [16384×K]@[K×N], decoder shapes |
| prefill GEMM, FP8 | | **98-152 TFLOPS** | `torch._scaled_mm` |

With these values the roofline predicts **2,290 tok/s at τ=1** and **3,480 tok/s at τ=1.65**
(32K context), up from 2,013 and 3,039. *Those numbers assume decode kernels that stream
weights at the measured 87% efficiency.* The run then showed our first kernel doesn't:

### Decode weight stream, kernel v0 (one launch per matrix, CUDA cores)

Replays all 111 matrices of a cirrus-s-1a decode step (90.8 MB NVFP4, random weights):

| tokens verified (M) | 1 | 2 | 4 | 8 | 13 |
|---|---|---|---|---|---|
| µs / step | 963 | 1,687 | 3,134 | 5,948 | 9,512 |
| effective GB/s | 94 | 54 | 29 | 15 | 9.5 |

cuBLAS BF16 on the same sequence: 1,815 µs at M=1 (178 GB/s), and almost flat in M (1,878 µs at M=8).

Diagnosis:
1. **M=1 is latency-bound, not bandwidth-bound.** Most matrices are only 0.3-2 MB, so each
   kernel finishes before the memory pipeline fills, and we pay launch + DRAM latency 111 times.
   This is why the design calls for one persistent kernel per step.
2. **Cost grows linearly with M because the kernel is instruction-bound.** Each extra token adds
   a multiply-add and an activation load *per weight* on CUDA cores. The weights are still read
   once. The fix is tensor cores, where one `mma.m16n8k16` covers 8 tokens.

### Other findings

- The L2 sweep showed L2-like speed up to 64 MB, well past the 24 MB L2. Those buffers were
  all-ones, so run 2 repeats the sweep with random data to tell a second cache level (or SoC
  cache) apart from data compression. If it is real, it matters: the whole NVFP4 model is ~87 MB.
- The reference model ran on the GPU for the first time. Incremental decode matches the full
  forward pass to 6.7e-6 (dense attention path).
- `test_incremental_decode_matches_full_forward[sparse]` failed. Cause: ReLU indexer scores are
  often exactly 0, and `topk` breaks ties differently in prefill rows (which contain masked
  future keys) than in decode rows. Fixed with a stable, most-recent-first tie-break
  (`cirrus/model/layers.py::topk_mask`), and a regression test was added.
- `flash-linear-attention` is not installed; the GDN reference path was used.

## Run 2 (2026-10-24): kernel v1

All kernels built, all correctness checks passed (mma GEMV 2.6e-7, persistent stream 3.0e-7),
and **all 35 tests passed**, including the sparse incremental-decode test fixed after run 1.

### Decode weight stream (90.8 MB, tensor-core NVFP4)

| M (tokens verified) | 1 | 2 | 4 | 8 | 13 |
|---|---|---|---|---|---|
| v0 per-matrix, CUDA cores (run 1) | 963 µs | 1,687 | 3,134 | 5,948 | 9,512 |
| v1 per-matrix, CUDA graph | 732 µs | 760 | 819 | 925 | 1,446 |
| **v1 persistent, 78 stages** | **638 µs (142 GB/s)** | 656 | 687 | 773 | 1,042 |

Tensor cores fixed the scaling with M: going from 1 to 8 tokens now costs +21%, not +518%. But
the persistent stream reaches only 52% of the DRAM peak (60% of the 238 GB/s streaming read),
which gives **~2,190 tok/s at τ=1.65 (verify 4)** and ~2,980 at τ=2.5 (verify 8).

Diagnosis:
- **≈3 µs of fixed cost per stage.** (638 − 381 µs ideal) / 78 stages ≈ 3.3 µs. Every stage
  waits for DRAM latency, the grid barrier and the activation load before its first weight
  bytes arrive. Weights don't depend on activations, so they can be in flight across the barrier.
- **Global split-K atomics cost ≈15 µs per million.** From the slopes: M 1→4 adds 50 µs,
  M 4→8 adds 86 µs. The atomic count scales with M (≈1.2M at M=1, ≈9.6M at M=8).

### The cache surprise is real (not compression)

| working set | 24 MB | 32 | 48 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| random data, GB/s | 2,049 | 2,028 | 2,002 | 1,972 | 972 | 292 | 235 |
| constant data, GB/s | 2,049 | 2,000 | 2,000 | 1,999 | 1,259 | 333 | 245 |

Random and constant data behave the same, so this isn't compression. Something serves
**≈64 MB at ≈2 TB/s** and part of 96 MB. Public material lists a 24 MB GPU L2 plus a 16 MB
system-level cache (40 MB total) ([Hot Chips 2025](https://www.hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf),
[Chips and Cheese](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu)), and the GPU's use of
the SLC isn't documented. That still doesn't explain a full-speed plateau at 64 MB, so run 3
adds a rotating-offset sweep and a pointer-chase latency curve to rule out any per-thread reuse
artifact.

**Why it matters:** the NVFP4 decode working set is ~91 MB, just past the edge. If ≈64 MB can
stay resident across decode steps, a model whose per-step weights fit (≈110M params at NVFP4,
or 155M at ~3 bits) could stream at several times DRAM speed.

## Run 3 (pending): kernel v2 + cache probes

`scripts/bench_spark.py --only decode_stream_v2 cache_resident_stream cache_probe model_smoke_test`

- **v2 pipelined persistent stream:** each warp always has its next chunk of weight loads in
  flight, across task *and* stage boundaries, so DRAM never idles at a barrier. Split-K is sized
  per stage (46,767 tasks for the full step versus ~89K chunks before), which also cuts atomics.
  The traversal was verified offline (every chunk computed once, every task flushed once).
- **Cache-policy variants:** evict-first (`.cs`), L2 (`.cg`) and read-only (`.nc`) loads.
- **Cache-resident test:** streams the first 16/32/48/64/80 MB of the model back to back.
- **Cache probes:** rotating-offset read sweep, and dependent-load latency vs. working set.
