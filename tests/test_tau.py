import pytest

from cirrus.decoding.tau import copy_fraction, simulate
from cirrus.perf.estimate import RUN6_STREAM_US, interp_stream_us, tok_s_from_stream


def test_no_overlap_lookup_only_is_one_token_per_step():
    st = simulate(list(range(100, 200)), list(range(1, 50)), max_draft=7)
    assert st.tau == 1.0 and st.steps == 48


def test_copied_span_is_drafted_in_long_runs():
    source = list(range(1000, 1300))
    out = source[100:220]
    st = simulate(source, out, max_draft=7)
    assert st.tokens == len(out) - 1
    assert st.tau > 6.0  # nearly every step accepts the full 7-token draft


def test_mtp_model_bounds():
    out = list(range(1, 61))
    perfect = simulate([999] * 10, out, mtp=(1.0, 1.0))
    assert abs(perfect.tau - 3.0) < 0.1  # 2 drafted + 1 verified every step
    none = simulate([999] * 10, out, mtp=(0.0, 0.0))
    assert none.tau == 1.0


def test_tree_never_worse_than_fallback_and_seeded():
    source = list(range(500, 800))
    out = source[10:40] + list(range(1, 30)) + source[200:240]
    fb = simulate(source, out, mtp=(0.8, 0.6), mode="fallback", seed=3)
    tr = simulate(source, out, mtp=(0.8, 0.6), mode="tree", seed=3)
    assert tr.tau >= fb.tau
    assert simulate(source, out, mtp=(0.8, 0.6), seed=3).tau == fb.tau
    with pytest.raises(ValueError):
        simulate(source, out, mode="bogus")


def test_copy_fraction():
    src = [1, 2, 3, 4, 5, 6, 7, 8]
    assert copy_fraction(src, [1, 2, 3, 4, 5]) == 1.0
    assert copy_fraction(src, [9, 9, 9, 9, 9]) == 0.0


def test_estimator_reproduces_run6_report():
    # bench_spark.py printed these with the same formula on 2026-10-24
    assert round(tok_s_from_stream(RUN6_STREAM_US, 1.65, 4, 32768)) == 2955
    assert round(tok_s_from_stream(RUN6_STREAM_US, 2.0, 4, 4096)) == 3632
    assert round(tok_s_from_stream(RUN6_STREAM_US, 2.5, 8, 32768)) == 4372
    assert interp_stream_us(RUN6_STREAM_US, 6) == pytest.approx((491.8 + 505.3) / 2)
