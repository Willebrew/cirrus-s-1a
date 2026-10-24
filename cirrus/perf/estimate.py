"""Decode tok/s from a *measured* weight-stream time.

The persistent-kernel benchmark measures the dominant part of a decode step:
streaming every weight matrix, with its barriers. The small remaining
traffic (GDN state, sliding-window and top-k KV, indexer keys) and the
per-step fixed costs still come from the roofline model.
"""

from __future__ import annotations

from cirrus.config import ModelConfig, get_config
from cirrus.perf.hardware import DGX_SPARK, HardwareSpec
from cirrus.perf.roofline import InferencePlan, decode_step

# Best measured stream per verify width M, kernel v5, Spark run 6 (2026-10-24,
# results/spark-grace-20261024-144653.json), microseconds.
RUN6_STREAM_US: dict[int, float] = {1: 490.3, 2: 485.7, 4: 491.8, 8: 505.3, 13: 625.9}

_WEIGHT_KEYS = ("weights.body", "lm_head", "mtp.weights", "mtp.draft_head")


def interp_stream_us(stream_us: dict[int, float], m: int) -> float:
    """Linear interpolation over the measured verify widths (clamped)."""
    ms = sorted(stream_us)
    if m <= ms[0]:
        return stream_us[ms[0]]
    if m >= ms[-1]:
        return stream_us[ms[-1]]
    for lo, hi in zip(ms, ms[1:]):
        if lo <= m <= hi:
            f = (m - lo) / (hi - lo)
            return stream_us[lo] + f * (stream_us[hi] - stream_us[lo])
    raise AssertionError("unreachable")


def tok_s_from_stream(
    stream_us: dict[int, float],
    tau: float,
    verify: int,
    context_len: int = 32768,
    cfg: ModelConfig | None = None,
    hw: HardwareSpec = DGX_SPARK,
) -> float:
    cfg = cfg or get_config("cirrus-s-1a")
    use_mtp = tau > 1
    s = decode_step(cfg, hw, InferencePlan(tokens_per_step=tau, use_mtp=use_mtp, n_verify=verify), context_len)
    other = s.total_dram_bytes - sum(s.dram_bytes.get(k, 0) for k in _WEIGHT_KEYS)
    t_other = other / hw.eff_dram_bw + sum(s.l2_bytes.values()) / hw.eff_l2_bw
    # mixer stages (GDN recurrence, attention) add ~1 barrier per layer; the
    # second MTP pass adds 4 stages; plus the host round trip
    barriers = cfg.n_layers + ((cfg.mtp_passes - 1) * 4 if use_mtp else 0)
    t_fixed = (barriers * hw.grid_barrier_us + hw.host_sync_us) * 1e-6
    step = interp_stream_us(stream_us, verify) * 1e-6 + t_other + t_fixed
    return tau / step
