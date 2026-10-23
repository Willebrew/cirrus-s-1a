"""Reference-model tests. Skipped when torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from cirrus.config import get_config  # noqa: E402
from cirrus.model.generate import greedy, speculative_greedy  # noqa: E402
from cirrus.model.layers import ShortConv, gated_delta_rule_ref  # noqa: E402
from cirrus.model.model import CirrusModel  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    # exact comparisons between decode paths need float64
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def model(**overrides):
    from dataclasses import replace

    return CirrusModel(replace(get_config("cirrus-s-1a-debug"), **overrides)).eval()


def test_param_count_matches_config():
    for name in ("cirrus-s-1a-debug",):
        cfg = get_config(name)
        assert CirrusModel(cfg).num_params() == cfg.total_params()


def test_gdn_chunked_state_equals_full():
    B, T, H, Dk, Dv = 2, 9, 2, 4, 3
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Dk), dim=-1)
    v, g, beta = torch.randn(B, T, H, Dv), -torch.rand(B, T, H), torch.rand(B, T, H)
    o_full, S_full = gated_delta_rule_ref(q, k, v, g, beta, 0.5)
    o1, S1 = gated_delta_rule_ref(q[:, :4], k[:, :4], v[:, :4], g[:, :4], beta[:, :4], 0.5)
    o2, S2 = gated_delta_rule_ref(q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:], 0.5, S1)
    torch.testing.assert_close(torch.cat((o1, o2), 1), o_full)
    torch.testing.assert_close(S2, S_full)


def test_short_conv_cache():
    conv = ShortConv(5, 4)
    x = torch.randn(1, 7, 5)
    y_full, _ = conv(x, None)
    y1, c = conv(x[:, :3], None)
    y2, _ = conv(x[:, 3:], c)
    torch.testing.assert_close(torch.cat((y1, y2), 1), y_full)


@pytest.mark.parametrize("sparse", [True, False])
def test_incremental_decode_matches_full_forward(sparse):
    m = model(sparse_topk=6 if sparse else None)
    ids = torch.randint(0, m.cfg.vocab_size, (1, 20))
    full = m(ids).logits
    out = m(ids[:, :11], use_cache=True)
    step = [out.logits]
    cache = out.cache
    for t in range(11, 20):
        o = m(ids[:, t : t + 1], cache=cache)
        step.append(o.logits)
        cache = o.cache
    torch.testing.assert_close(torch.cat(step, 1), full)


def test_yoco_prefill_shortcut_gives_same_last_logits():
    m = model()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    torch.testing.assert_close(m(ids, n_logits=1).logits[:, -1], m(ids).logits[:, -1])


def test_loss_backward_with_mtp_and_indexer():
    m = model().train()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 24))
    labels = ids.clone()
    labels[:, :5] = -100  # e.g. harness-written tool results
    out = m(ids, labels=labels, indexer_weight=0.1)
    assert {"ce", "mtp", "indexer_kl"} <= set(out.metrics)
    out.loss.backward()
    assert m.shared.idx_k.weight.grad is not None
    assert m.mtp[0].proj.weight.grad is not None


def test_speculative_greedy_is_token_identical_to_greedy():
    m = model()
    V = m.cfg.vocab_size
    base = torch.randint(0, V, (12,)).tolist()
    prompt = base + torch.randint(0, V, (5,)).tolist() + base[:4]  # repeated span to copy
    ref, s_ref = greedy(m, prompt, 30)
    spec, s_spec = speculative_greedy(m, prompt, 30)
    assert spec == ref
    assert s_spec.steps <= s_ref.steps
