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

## Run 3 (2026-10-24): kernel v2 + cache probes

Built cleanly; all checks passed (v2 stream 3.5e-7); 35/35 tests.

**v2 was slower than v1:** 696 µs at M=1 with its best hint (evict-first), versus 638 µs for v1.
The L2 hint (`.cg`) was 4-7% slower on the full 91 MB step.

**The cache-resident test shows the real problem.** Even a 15.6 MB, 15-stage slice of the
model, which fits in L2, streamed at only 194 GB/s (`.cg`) versus 1,864 GB/s for a plain read
of the same size: about **5.4 µs per stage with no DRAM involved**. The kernel is
latency-bound in its own control path:
1. **Dependent global loads per chunk.** v2 reads the task table, then the matrix descriptor,
   then issues the weight load: two ~140 ns round trips (pointer-chase latency) before the
   prefetch even starts, and the warp stalls on them ahead of its math.
2. **`load_x` is serialized.** 6-16 iterations per thread, each store waiting on its load:
   ≈1-2 µs per stage.
3. **Dequant instruction count** (~61 instructions per 32-bit weight word) lengthens each
   stage's critical path.

**The cache is about 32 MB, not 64 MB.** The rotating-offset read holds 1.65-1.86 TB/s up to
32 MB, then drops to 718 GB/s at 40 MB and DRAM speed by 80 MB. That's consistent with the
documented 24 MB L2 plus 16 MB SLC. The plain sweep's 64 MB plateau was flattered by
per-thread reuse. Dependent-load latency is a flat ~140 ns from 0.25 MB to 256 MB, so on this
part an L2 hit and a DRAM access cost about the same *latency*; caches help bandwidth, not
latency. That's another reason latency chains are so costly here.

**Consequences for the design:**
- Cache residency of all the weights is off the table at any sensible size, but the L2 hint
  does pay off for data that truly fits (16 MB slice: 194 vs 140 GB/s). It suits the shared
  KV/top-k rows and the MTP head, which the roofline already assumes are L2-resident.
- Per-stage latency is the critical engineering problem. The real engine has ~95 dependent
  stages per step (norms and activations fold into activation loads, plus GDN/attention cores).

## Run 4 (2026-10-24): kernel v3, first pass over 3,000 tok/s

Built cleanly; v3 correctness 4.1e-7; 35/35 tests.

| M | 1 | 2 | 4 | 8 | 13 |
|---|---|---|---|---|---|
| v1 persistent (run 2) | 638 µs | 656 | 687 | 773 | 1,042 |
| **v3, `.cs`** | **546 µs (166 GB/s)** | 554 | 572 | 567 | 748 |
| v3, `.cg` | 563 | 576 | 594 | 631 | 824 |
| v3, `.cs` **no math** | 547 | 555 | 565 | 582 | 722 |

**Decode estimate from the measured stream: 3,131 tok/s at τ=2.0 (verify 4) and 3,945 at
τ=2.5 (verify 8), 32K context. That's the first configuration over 3,000.** τ=1.65 gives 2,583.

**Arithmetic is no longer the limit:** removing all dequant and tensor-core math changes the
time by <1% for M ≤ 4.

Per-stage profile (M=1, `clock64` stamps): 563.7 µs = 96.2 µs activation load + 424.7 µs
compute (slowest SM) + 50.5 µs barrier/imbalance.
- **Big stages stream near peak:** `lm_head` (14.2 MB) at ~234 GB/s during compute, and
  `ffn.gate_up` (1.77 MB) at ~221 GB/s.
- **Stage boundaries cost ≈1.7 µs each** (1.06 µs activation load + 0.65 µs barrier), ≈133 µs per
  step. (546 − 133) µs × ~225 GB/s ≈ 93 MB, i.e. DRAM is effectively idle during the gaps.
- **Small stages run on 18 of 48 SMs.** For the 0.33 MB output projections, the median block's
  compute is 0.07 µs: warp-major task placement put their 288 tasks on the first 18 SMs. An
  offline emulation reproduces exactly this.

## Run 5 (2026-10-24): kernel v4, placement × prefetch depth

Built cleanly; v4 correctness 3.8e-7 across all depths and placements; 35/35 tests.

| M | 1 | 2 | 4 | 8 | 13 |
|---|---|---|---|---|---|
| v3 `.cs` (run 4) | 546 µs | 554 | 572 | 567 | 748 |
| **v4 best** | **512 µs (177.5 GB/s)** | 514 | 519 | 526 | 651 |
| best config | wm d2 | rr d1 | wm d1 | wm d1 | rr d2 |

All configurations are within ~3% of each other up to M=8. **Decode estimate from the measured
stream: 2,819 tok/s at τ=1.65, 3,417 at τ=2.0, 4,222 at τ=2.5 (32K context).** τ=1.65 is now
within ~6% of the target, and 3,000 tok/s is reached at τ ≈ 1.75.

Profile (M=1, wm d2): 528 µs = **99 µs activation load + 393 µs compute + 50 µs barrier/
imbalance**. The gain over v3 came entirely from the compute phase (425 → 393 µs). FFN
up-projections now finish 1.77 MB in ~7.1 µs of compute, *faster* than DRAM streaming
(~249 GB/s), because prefetched weights arrive before the stage starts. The fixed costs didn't
move:
- **Activation load: a flat 1.06 µs per stage at K=768 (1.73 µs at K=2048), even at M=1**,
  where only a single 1.5 KB row is real. The cost tracks the loader's per-thread work (integer
  divisions over all 512 threads, plus zero-filling 7 unused token rows), not bytes.
- **Barrier + imbalance: ~0.65 µs per stage**, close to the bare 0.61 µs `grid.sync()`.
- With depth 2, the first stage's activation load waits 7.4 µs behind the prologue's weight
  loads (2.35 µs at depth 1). That's a one-time cost per step.

## Run 6 (pending): kernel v5, real-rows-only activation loader

`scripts/bench_spark.py --only decode_stream_v5 model_smoke_test`

- `load_x4` loads only the M real token rows, with a division-free thread mapping (2 rows per
  pass, 256 threads per row). Rows ≥ M are deliberately left stale: in `mma` each output column
  depends only on its own activation column, so stale rows reach only outputs that are never
  stored. The correctness check alternates K between stages so stale rows are actually present.
- Swept against the old loader on the same configurations; the best is profiled per stage.
- Target: activation load ~0.3-0.4 µs per stage (≈ one L2 round trip + `__syncthreads`), about
  50 µs off the step, putting τ=1.65 over 3,000 tok/s.
- Known gap in the estimator: the 2nd MTP draft pass assumes its 6.9 MB of weights hit L2,
  but the stream loads them evict-first. The real engine should load MTP weights with the L2
  hint.
