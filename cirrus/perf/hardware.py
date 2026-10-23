"""Hardware descriptions used by the roofline model.

Every number here carries a source. Values marked ``ESTIMATE`` are still
engineering guesses; the rest were measured with ``scripts/bench_spark.py``
(see docs/research/spark-measurements.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    # DRAM
    dram_bw_gbps: float               # theoretical peak, GB/s
    dram_efficiency: float            # achievable fraction for streaming GEMV
    memory_gb: float
    # On-chip
    n_sms: int
    l2_bytes: int
    l2_bw_gbps: float                 # ESTIMATE unless measured
    smem_per_sm_bytes: int
    # Dense tensor-core throughput, TFLOPS
    tflops_fp4: float
    tflops_fp8: float
    tflops_bf16: float
    tensor_efficiency: float          # achievable fraction for prefill GEMMs
    # Fixed per-step costs, microseconds
    grid_barrier_us: float            # one grid-wide sync inside a persistent kernel
    kernel_launch_us: float           # per kernel node in a CUDA graph
    host_sync_us: float               # GPU->CPU->GPU round trip per decode step

    @property
    def eff_dram_bw(self) -> float:
        """Achievable bytes/second."""
        return self.dram_bw_gbps * 1e9 * self.dram_efficiency

    @property
    def eff_l2_bw(self) -> float:
        return self.l2_bw_gbps * 1e9


# NVIDIA DGX Spark, GB10 Grace Blackwell superchip.
# MEASURED on "grace" (DGX OS 7.2.3, driver 580.178.04, CUDA 13.0, torch
# 2.13.0+cu130) by scripts/bench_spark.py on 2026-10-23; raw data in
# results/spark-grace-20261023-050307.json. Peak figures are NVIDIA's spec.
DGX_SPARK = HardwareSpec(
    name="DGX Spark (GB10)",
    dram_bw_gbps=273.0,         # spec
    dram_efficiency=0.873,      # measured: 238.4 GB/s streaming read
    memory_gb=128.0,
    n_sms=48,                   # measured
    l2_bytes=24 * 2**20,        # measured (cudaDeviceProp.l2CacheSize)
    l2_bw_gbps=2043.0,          # measured plateau, working sets <= 24 MB
    smem_per_sm_bytes=100 * 1024,
    tflops_fp4=300.0,           # ESTIMATE: ~2x measured FP8; not yet measured
    tflops_fp8=152.0,           # measured best on decoder-shaped prefill GEMMs
    tflops_bf16=93.0,           # measured best on decoder-shaped prefill GEMMs
    tensor_efficiency=1.0,      # the tflops above are achieved, not peak
    grid_barrier_us=0.61,       # measured: cooperative grid.sync(), 48 blocks
    kernel_launch_us=0.94,      # measured: CUDA graph node
    host_sync_us=12.5,          # measured: GPU->CPU->GPU from Python
)

HARDWARE: dict[str, HardwareSpec] = {"dgx-spark": DGX_SPARK}
