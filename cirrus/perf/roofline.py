"""Analytical roofline model for cirrus-s-1a decode and prefill on DGX Spark.

Decode on GB10 is memory-bandwidth bound: every forward step streams the
weights plus whatever KV/state it touches. The model therefore counts bytes
moved per step, converts them to time with the achievable bandwidth, adds
fixed synchronization overhead, and divides the expected number of accepted
tokens per step (speculative decoding) by that time.

    tok/s = tokens_per_step / (max(t_mem, t_compute) + t_fixed)

This is a planning tool, not a promise: every ESTIMATE in the hardware spec
must be replaced with measurements (scripts/bench_spark.py) before we trust
the absolute numbers. Relative comparisons between designs are robust.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from cirrus.config import ModelConfig
from cirrus.perf.hardware import HardwareSpec

# Bits per parameter including block scales.
# NVFP4: 4-bit E2M1 values + one FP8 (E4M3) scale per 16 values -> 4.5 bits.
# MXFP4: 4-bit + one E8M0 scale per 32 values -> 4.25 bits.
# FP8 with per-channel scales: ~8 bits.
BITS = {"nvfp4": 4.5, "mxfp4": 4.25, "fp8": 8.0, "bf16": 16.0}


@dataclass(frozen=True)
class InferencePlan:
    body_format: str = "nvfp4"           # transformer/GDN/FFN weights
    head_format: str = "nvfp4"           # tied embedding / LM head
    mtp_format: str = "nvfp4"
    kv_bits: int = 8                     # FP8 KV cache
    gdn_state_bytes: int = 2             # bf16 recurrent state (fp32 accumulate in regs)

    attention: Literal["dense", "sparse"] = "sparse"
    sparse_topk: int = 2048              # tokens each cross layer reads
    indexer_dim: int = 64                # FP8 indexer key per cached token

    # Fraction of L2 we allow the planner to pin (cudaAccessPolicyWindow).
    l2_pin_fraction: float = 0.6

    launch: Literal["megakernel", "cuda_graph"] = "megakernel"
    syncs_per_layer: int = 3             # grid barriers per layer in the megakernel
    kernels_per_layer: int = 7           # kernel nodes per layer under CUDA graphs
    host_sync_per_step: bool = True      # CPU-side drafting (prompt lookup) each step

    # Speculative decoding
    use_mtp: bool = True
    n_verify: int = 4                    # tokens checked per step (1 + drafts)
    tokens_per_step: float = 1.0         # expected accepted tokens per step


@dataclass
class StepCost:
    dram_bytes: dict[str, float] = field(default_factory=dict)
    l2_bytes: dict[str, float] = field(default_factory=dict)
    flops: float = 0.0
    t_mem_us: float = 0.0
    t_compute_us: float = 0.0
    t_fixed_us: float = 0.0
    tokens_per_step: float = 1.0

    @property
    def total_dram_bytes(self) -> float:
        return sum(self.dram_bytes.values())

    @property
    def step_us(self) -> float:
        return max(self.t_mem_us, self.t_compute_us) + self.t_fixed_us

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_per_step / (self.step_us * 1e-6)

    @property
    def bound(self) -> str:
        return "memory" if self.t_mem_us >= self.t_compute_us else "compute"


def _bytes(params: float, fmt: str) -> float:
    return params * BITS[fmt] / 8.0


class _L2:
    """Greedy L2 pinning: items that are read more than once per step and fit
    in the pinned budget are pulled from DRAM once and then served from L2."""

    def __init__(self, budget: float):
        self.budget = budget

    def place(self, c: "StepCost", key: str, nbytes: float, reads: int) -> None:
        if reads <= 0 or nbytes <= 0:
            return
        if reads > 1 and nbytes <= self.budget:
            self.budget -= nbytes
            c.dram_bytes[key] = nbytes
            c.l2_bytes[key] = nbytes * (reads - 1)
        else:
            c.dram_bytes[key] = nbytes * reads


def decode_step(
    cfg: ModelConfig,
    hw: HardwareSpec,
    plan: InferencePlan,
    context_len: int,
) -> StepCost:
    """Cost of one decode/verify forward at ``context_len`` cached tokens.

    One step = main model verifies ``n_verify`` positions in a single pass
    (weights streamed once), then the MTP module runs ``mtp_passes`` times to
    draft the next step's candidates. Prompt-lookup drafts cost nothing on the
    GPU; they only change ``tokens_per_step``.
    """
    c = StepCost(tokens_per_step=plan.tokens_per_step)
    kv_b = plan.kv_bits / 8.0
    kinds = cfg.layer_kinds
    n_gdn = kinds.count("gdn")
    n_swa = kinds.count("swa")
    n_cross = kinds.count("cross")
    mtp_passes = cfg.mtp_passes * cfg.n_mtp_heads if plan.use_mtp else 0
    n_tok = plan.n_verify if (plan.use_mtp or plan.tokens_per_step > 1) else 1
    l2 = _L2(hw.l2_bytes * plan.l2_pin_fraction)

    # ---------------------------------------------- shared (YOCO) global KV
    # Highest L2 priority: re-read by every cross layer.
    kv_row = 2 * cfg.kv_dim * kv_b
    if plan.attention == "dense":
        l2.place(c, "cross.kv", context_len * kv_row, n_cross)
    else:
        c.dram_bytes["cross.indexer"] = context_len * plan.indexer_dim * kv_b
        # one shared top-k selection per step (the KV cache is shared, so the
        # selection can be too); gathered rows then serve all cross layers
        l2.place(c, "cross.kv_topk", min(context_len, plan.sparse_topk) * kv_row, n_cross)

    # ------------------------------------------------------- draft module
    if mtp_passes:
        l2.place(c, "mtp.weights", _bytes(cfg.n_mtp_heads * cfg.mtp_head_params(), plan.mtp_format), mtp_passes)
        draft_head = _bytes(min(cfg.mtp_draft_vocab, cfg.vocab_size) * cfg.d_model, plan.head_format)
        l2.place(c, "mtp.draft_head", draft_head, mtp_passes)

    # ---------------------------------------------------------- weights
    c.dram_bytes["weights.body"] = _bytes(cfg.backbone_params(), plan.body_format)
    c.dram_bytes["lm_head"] = _bytes(cfg.embedding_params(), plan.head_format)

    # ---------------------------------------------------- recurrent state
    state = cfg.gdn_heads * cfg.gdn_head_dim_k * cfg.gdn_head_dim_v * plan.gdn_state_bytes
    c.dram_bytes["gdn.state_rw"] = 2 * state * n_gdn

    # ----------------------------------------------------- sliding window
    swa_tokens = min(context_len, cfg.swa_window)
    c.dram_bytes["swa.kv"] = n_swa * swa_tokens * 2 * cfg.kv_dim * kv_b

    # ------------------------------------------------------------- flops
    c.flops = 2.0 * (cfg.backbone_params() + cfg.embedding_params()) * n_tok
    if mtp_passes:
        c.flops += 2.0 * mtp_passes * (cfg.mtp_head_params() + cfg.mtp_draft_vocab * cfg.d_model)
    attn_ctx = min(context_len, plan.sparse_topk) if plan.attention == "sparse" else context_len
    c.flops += n_tok * n_cross * 4.0 * attn_ctx * cfg.q_dim
    c.flops += n_tok * n_swa * 4.0 * swa_tokens * cfg.q_dim
    if plan.attention == "sparse":
        c.flops += n_tok * 2.0 * context_len * plan.indexer_dim * cfg.n_q_heads

    # ------------------------------------------------------------ timing
    c.t_mem_us = (c.total_dram_bytes / hw.eff_dram_bw + sum(c.l2_bytes.values()) / hw.eff_l2_bw) * 1e6
    tensor = hw.tflops_fp4 if plan.body_format in ("nvfp4", "mxfp4") else hw.tflops_fp8
    # decode GEMMs have M = n_tok < 16, so tensor-core tiles run mostly padded
    pad = 16.0 / max(1, min(n_tok, 16))
    c.t_compute_us = c.flops * pad / (tensor * 1e12 * hw.tensor_efficiency) * 1e6

    head_passes = 1 + mtp_passes
    sequential_blocks = cfg.n_layers + mtp_passes
    if plan.launch == "megakernel":
        fixed = sequential_blocks * plan.syncs_per_layer * hw.grid_barrier_us
        fixed += head_passes * 2 * hw.grid_barrier_us  # logits + argmax/sample
    else:
        fixed = sequential_blocks * plan.kernels_per_layer * hw.kernel_launch_us
        fixed += head_passes * 2 * hw.kernel_launch_us
    if plan.host_sync_per_step:
        fixed += hw.host_sync_us
    c.t_fixed_us = fixed
    return c


@dataclass
class PrefillCost:
    flops: float
    t_us: float
    tokens: int


def prefill(cfg: ModelConfig, hw: HardwareSpec, plan: InferencePlan, n_tokens: int) -> PrefillCost:
    """Compute-bound prefill of ``n_tokens`` (e.g. freshly retrieved results).

    YOCO lets prompt tokens skip the cross-decoder: we only need the shared KV
    (and indexer keys) for them, so the cross layers run for the final token
    only, which is accounted for by the next decode step.
    """
    kinds = cfg.self_layer_kinds
    self_params = sum(cfg.layer_params(k) for k in kinds) + cfg.shared_kv_params()
    gemm = 2.0 * self_params * n_tokens
    attn = 0.0
    for k in kinds:
        if k == "swa":
            w = min(n_tokens, cfg.swa_window)
            attn += 4.0 * n_tokens * w * cfg.q_dim
        elif k == "gdn":
            chunk = 64
            per_tok_head = 4.0 * cfg.gdn_head_dim_k * cfg.gdn_head_dim_v + 2.0 * chunk * (
                cfg.gdn_head_dim_k + cfg.gdn_head_dim_v
            )
            attn += n_tokens * cfg.gdn_heads * per_tok_head
    gemm_t = hw.tflops_fp4 if plan.body_format in ("nvfp4", "mxfp4") else hw.tflops_fp8
    t = gemm / (gemm_t * 1e12 * hw.tensor_efficiency) + attn / (hw.tflops_bf16 * 1e12 * hw.tensor_efficiency)
    return PrefillCost(flops=gemm + attn, t_us=t * 1e6, tokens=n_tokens)


def expected_tokens_per_step(accept_probs: list[float]) -> float:
    """Expected accepted tokens for a chain of drafts with per-position
    conditional acceptance probabilities (plus the always-correct bonus token).
    """
    total, run = 1.0, 1.0
    for p in accept_probs:
        run *= p
        total += run
    return total


def with_tokens_per_step(plan: InferencePlan, tps: float) -> InferencePlan:
    return replace(plan, tokens_per_step=tps)
