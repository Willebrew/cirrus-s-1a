"""Hardware descriptions used by the roofline model.

Every number here should carry a source in docs/research/dgx-spark.md. Values
marked ``ESTIMATE`` are engineering guesses that must be replaced by
measurements from ``scripts/bench_spark.py`` once we have a Spark in hand.
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
# Confirmed: 128 GB LPDDR5x unified, 273 GB/s, 6144 CUDA cores (48 SMs),
# 1 PFLOP sparse FP4. See docs/research/dgx-spark.md for sources and the
# remaining ESTIMATE fields.
DGX_SPARK = HardwareSpec(
    name="DGX Spark (GB10)",
    dram_bw_gbps=273.0,
    dram_efficiency=0.80,       # ESTIMATE: LPDDR5x streaming, refine w/ nvbandwidth
    memory_gb=128.0,
    n_sms=48,
    l2_bytes=24 * 2**20,        # ESTIMATE pending confirmation
    l2_bw_gbps=1500.0,          # ESTIMATE
    smem_per_sm_bytes=100 * 1024,
    tflops_fp4=500.0,           # 1 PFLOP sparse -> ~500 dense
    tflops_fp8=250.0,           # ESTIMATE (half of FP4 dense)
    tflops_bf16=125.0,          # ESTIMATE (half of FP8 dense)
    tensor_efficiency=0.45,     # ESTIMATE: small-model GEMMs rarely saturate
    grid_barrier_us=1.0,        # ESTIMATE: cooperative-groups grid.sync on 48 SMs
    kernel_launch_us=2.5,       # ESTIMATE: graph node + tail effect
    host_sync_us=15.0,          # ESTIMATE: unified memory, no PCIe
)

HARDWARE: dict[str, HardwareSpec] = {"dgx-spark": DGX_SPARK}
