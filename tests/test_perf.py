from dataclasses import replace

from cirrus.config import PRESETS, get_config
from cirrus.perf.hardware import DGX_SPARK
from cirrus.perf.roofline import InferencePlan, decode_step, expected_tokens_per_step, prefill


def test_param_counts_are_in_target_range():
    assert 250e6 < get_config("cirrus-s-1a").total_params() < 350e6
    assert 120e6 < get_config("cirrus-s-1a-mini").total_params() < 200e6
    for cfg in PRESETS.values():
        assert len(cfg.layer_kinds) == cfg.n_layers


def test_sparse_attention_makes_decode_nearly_context_independent():
    cfg = get_config("cirrus-s-1a")
    plan = InferencePlan(attention="sparse")
    short = decode_step(cfg, DGX_SPARK, plan, 4096).step_us
    long = decode_step(cfg, DGX_SPARK, plan, 65536).step_us
    assert long / short < 1.05
    dense = decode_step(cfg, DGX_SPARK, replace(plan, attention="dense"), 65536).step_us
    assert dense > 1.5 * long


def test_decode_is_memory_bound_and_spec_decoding_scales():
    cfg = get_config("cirrus-s-1a")
    one = decode_step(cfg, DGX_SPARK, InferencePlan(use_mtp=False), 32768)
    three = decode_step(cfg, DGX_SPARK, InferencePlan(tokens_per_step=3.0), 32768)
    assert one.bound == "memory"
    assert three.tokens_per_second > 2.5 * one.tokens_per_second


def test_megakernel_beats_cuda_graphs():
    cfg = get_config("cirrus-s-1a")
    mk = decode_step(cfg, DGX_SPARK, InferencePlan(launch="megakernel"), 8192)
    cg = decode_step(cfg, DGX_SPARK, InferencePlan(launch="cuda_graph"), 8192)
    assert mk.t_fixed_us < cg.t_fixed_us


def test_expected_tokens_per_step():
    assert expected_tokens_per_step([]) == 1.0
    assert abs(expected_tokens_per_step([0.5, 0.5]) - 1.75) < 1e-9


def test_prefill_scales_linearly_ish():
    cfg = get_config("cirrus-s-1a")
    a = prefill(cfg, DGX_SPARK, InferencePlan(), 4096).t_us
    b = prefill(cfg, DGX_SPARK, InferencePlan(), 16384).t_us
    assert 3.5 < b / a < 5.0
