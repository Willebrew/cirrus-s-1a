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

## Run 2 (pending): kernel v1

`scripts/bench_spark.py --only l2_sweep decode_stream_v1 model_smoke_test`, then `pytest -q`

- **Tensor-core NVFP4 GEMV:** weights pre-shuffled into a fragment-major tile layout (one
  16-byte load per lane per 16×64 tile), exact FP4→FP16 bit-trick dequant, `mma.m16n8k16` with
  FP32 accumulation, split-K over 128-wide chunks. Checked against a dense reference. The
  layout↔fragment mapping was also verified offline in a numpy emulation of the kernel.
- **Persistent stream kernel:** the megakernel's skeleton. One cooperative launch streams all
  78 fused stages of a decode step (matrices sharing an input form one stage) with `grid.sync()`
  between stages.
- **New estimate:** decode tok/s computed from the *measured* persistent stream, with only the
  small KV/state traffic from the roofline.
