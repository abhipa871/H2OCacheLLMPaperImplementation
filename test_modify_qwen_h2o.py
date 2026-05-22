"""
CPU: python -m pytest test_modify_qwen_h2o.py -m 'not gpu'
GPU: python -m pytest test_modify_qwen_h2o.py -m gpu

Synthetic + HF integration checks for ``modify_qwen``: padding-aware validity masks,
causal addends, HH-score aggregation, eviction with/without masks, attention scaling parity,
and (GPU) batched decode invariants vs ``past_key_values.h2o_next_position``.
"""

import copy
import math
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
import torch

_D = Path(__file__).resolve().parent
if str(_D) not in sys.path:
    sys.path.insert(0, str(_D))

from modify_qwen import (  # noqa: E402
    H2OKVCache,
    H2OQwen3_5Attention,
    H2OQwen3_5ForCausalLM,
    build_h2o_kv_valid_mask,
    make_h2o_causal_mask,
)


class KVPast:
    """Lightweight substitute for transformers ``past_key_values`` tuples in eviction unit tests.

    ``__getitem__(0)/(1)`` expose keys/values tensors; ``h2o_next_position`` mimics DynamicCache hooks
    when written by ``H2OKVCache``. Note ``evict_for_space`` returns new tensors—it does not always
    alias ``past[0]`` in-place.
    """

    __slots__ = ("h2o_next_position", "_k", "_v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.h2o_next_position = None
        self._k = k
        self._v = v

    def __getitem__(self, i: int):
        return self._k if i == 0 else self._v


def pytest_configure(config):
    """Register custom markers so selects like ``pytest -m gpu`` do not emit unknown-marker warnings."""

    config.addinivalue_line("markers", "gpu: CUDA + HF")


@pytest.fixture(scope="session")
def qwen_cfg():
    """Session-wide fixture: HF ``AutoConfig`` for ``Qwen/Qwen3.5-0.8B`` (``trust_remote_code``).

    Cloned inside tests where mutable fields (``hh_size`` / ``recent_size``) are patched.
    """

    from transformers import AutoConfig

    return AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)


def _ninf(dtype):
    """Return ``torch.finfo(dtype).min`` (largest-magnitude negative finite) for blocked-mask value checks."""

    return torch.finfo(dtype).min


def test_kv_valid_inside_bounds_reflects_gathered_padding_row():
    """``build_h2o_kv_valid_mask`` gathers ``attention_mask`` columns indexed by KV slot ids.

    Setup: padded row has ``False`` for cols 0--5 and ``True`` from 6 onward; KV positions probe 4..7
    replicated across ``nk`` heads. Expect cols 4--5 (still pad under this mask) map to validity ``False``
    everywhere; cols 6--7 map to ``True``.
    """

    B, nk, L = 1, 2, 8
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[:, 6:] = True
    kv_pid = torch.tensor([[4, 5, 6, 7]], dtype=torch.long).view(B, 1, -1).expand(B, nk, -1)
    vm = build_h2o_kv_valid_mask(kv_pid, attn)
    assert (~vm[..., :2]).all() and vm[..., 2:].all()


def test_kv_valid_oob_assumes_generation_valid():
    """Out-of-mask positions (``kv_position_ids >= attention_mask.width``) must count as valid KV slots.

    Each batch row mixes an in-bound index with a huge offset (simulating continued decode past the
    padded prompt). Second column is always ``True`` in the returned mask (implements “generated token” path).
    """

    B, nk, L = 2, 1, 6
    attn = torch.ones(B, L, dtype=torch.bool)
    kv_pid = torch.tensor([[[L - 1, L + 100]], [[0, L + 999]]])
    vm = build_h2o_kv_valid_mask(kv_pid, attn)
    assert vm[:, :, 1].all()


def test_causal_masks_where_kv_strictly_after_any_query_blocked():
    """``make_h2o_causal_mask`` should emit ``-inf`` wherever no query position can attend to a KV slot.

    Multi-query case: if any query ``q`` has all KV keys strictly after ``q``, those KV columns are fully
    masked. Uses a dummy all-``True`` padding mask so only causality is under test; repeats to full Q head count.
    """

    B, nk, _, kvlen = 1, 2, 2, 5
    q_arange = torch.tensor([[9, 11]], dtype=torch.long)
    kv_arange = torch.tensor([[[8, 9, 10, 11, 12]]], dtype=torch.long).expand(B, nk, kvlen)
    nh = nk * 6
    dtype = torch.float32
    m = make_h2o_causal_mask(
        q_arange,
        kv_arange,
        torch.ones(B, 256, dtype=torch.bool),
        num_attention_heads=nh,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    causal = kv_arange[:, :, None, :].repeat_interleave(nh // nk, dim=1) <= q_arange[:, None, :, None]
    forbid = torch.where(~causal, m, torch.zeros_like(m))
    assert torch.all(forbid[~causal] == _ninf(dtype))


def test_causal_plus_padding_blocked():
    """Batched causal + padding combination: additive mask must be ``-inf`` unless both causal and pad-valid.

    Two rows with different real-token spans; ``q_arange`` / ``kv_arange`` encode absolute positions.
    Cross-checks against ``build_h2o_kv_valid_mask`` expanded to attention head count.
    """

    B, nk, _, kl = 2, 2, 1, 5
    L = 96
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[0, 90:] = True
    attn[1, 88:] = True
    q_arange = torch.tensor([[92], [91]], dtype=torch.long)
    rows = [
        torch.arange(82, 82 + kl, dtype=torch.long).view(1, 1, -1),
        torch.arange(80, 80 + kl, dtype=torch.long).view(1, 1, -1),
    ]
    kv_arange = torch.cat(rows, dim=0).expand(-1, nk, -1)
    nh = 8
    dtype = torch.float64
    m = make_h2o_causal_mask(
        q_arange,
        kv_arange,
        attn,
        num_attention_heads=nh,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    vm = build_h2o_kv_valid_mask(kv_arange, attn).repeat_interleave(nh // nk, dim=1)
    causal = kv_arange[:, :, None, :].repeat_interleave(nh // nk, dim=1) <= q_arange[:, None, :, None]
    allowed = causal & vm[:, :, None, :]
    assert torch.all(torch.where(~allowed, m, torch.zeros_like(m)) == _ninf(dtype))


def test_score_gqa_group_sum():
    """First ``H2OKVCache._update_hh_score`` call on GQA-shaped attention weights.

    ``z`` is ``(B, n_attn_heads, q_len, kv_len)``; implementation should fold group/query dims so
    ``hh_score`` matches summing over query tokens and within each KV head group (here ``nk=2``, ``grp=2``).
    """

    nk, grp, qt, ks = 2, 2, 2, 17
    c = H2OKVCache(
        hh_size=1,
        recent_size=1,
        layer_idx=0,
        num_attention_heads=nk * grp,
        num_key_value_heads=nk,
        num_key_value_groups=grp,
    )
    z = torch.randn(1, nk * grp, qt, ks)
    c._update_hh_score(z)
    gold = z.view(1, nk, grp, qt, ks).sum(dim=(2, 3))
    assert torch.allclose(c.hh_score, gold)


def test_score_carry_prefix_two_calls():
    """Two-step HH score recurrence: old prefix is added to overlapping KV columns, new query columns start fresh.

    ``step1`` seeds ``hh_score``; ``step2`` extends ``kv_len``—asserts carry on shared prefix and direct assignment
    on the tail segment (non-GQA path, single KV head).
    """

    c = H2OKVCache(
        hh_size=2,
        recent_size=2,
        layer_idx=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_key_value_groups=1,
    )
    step1 = torch.tensor([[[[3.0, 1.0, 900.0]]]])
    c._update_hh_score(step1)
    carry = c.hh_score.clone()
    step2 = torch.tensor(
        [
            [
                [
                    [2.0, 4.0, 44.0, 66.0, 0.01],
                    [7.0, 8.0, 77.0, 88.0, 0.02],
                ]
            ]
        ]
    )
    summed = step2.sum(dim=2)
    qt = step2.shape[2]
    c._update_hh_score(step2)
    pref = summed.shape[-1] - qt
    assert pref == carry.shape[-1]
    assert torch.allclose(c.hh_score[:, :, :pref], summed[:, :, :pref] + carry)
    assert torch.allclose(c.hh_score[:, :, pref:], summed[:, :, pref:])


def test_eviction_recent_and_topkh():
    """``evict_for_space`` without padding mask: top-``hh_size`` from prefix + last ``recent_size`` slots.

    Keys embed slot index as float; ``hh_score`` spikes two prefix positions and a ramp—expects deterministic
    keep set after compressing to ``hh_size + recent_size`` (returned tensor ``[0]`` rows).
    """

    B, nh_kv, Dh = 1, 2, 6
    seq = 13
    hh_sz, rz = 2, 4
    keys = torch.empty(B, nh_kv, seq, Dh)
    for t in range(seq):
        keys[:, :, t] = float(t)
    vals = keys.clone()
    cache = H2OKVCache(
        hh_size=hh_sz,
        recent_size=rz,
        layer_idx=0,
        num_attention_heads=8,
        num_key_value_heads=nh_kv,
        num_key_value_groups=4,
    )
    cache.position_ids = torch.arange(seq).view(1, 1, -1).expand(B, nh_kv, -1).long()
    ph = seq - rz
    s = torch.zeros(B, nh_kv, seq)
    s[..., :ph] += torch.linspace(0.0, 10.0, ph).reshape(1, 1, -1)
    s[..., ph - 1] += 990.0
    s[..., ph - 3] += 980.0
    cache.hh_score = s.clone()
    vp = KVPast(keys.clone(), vals.clone())
    new_k = cache.evict_for_space(vp, num_coming=4096)[0].clone()
    assert new_k.shape[2] == hh_sz + rz
    want = sorted([ph - 3, ph - 1] + list(range(ph, seq)))
    assert sorted(new_k[0, 0, :, 0].round().long().tolist()) == want


def test_evict_prefers_valid_prefix_over_padding():
    """Padding-aware eviction: low ``hh_score`` on pad columns even if numerically large must not win top-k.

    ``attention_mask`` marks columns 0--5 pad; real span boosts scores around columns 7--9—after eviction the
    ``hh_size`` heads should be subsets of reals and the trailing window must be the unscored suffix.
    """

    B, nk, seq, Dh = 1, 1, 12, 4
    hh_sz, rz = 3, 2
    attn = torch.zeros(B, seq, dtype=torch.bool)
    attn[:, 6:] = True
    pref = seq - rz
    keys = torch.arange(seq, dtype=torch.float32).reshape(1, 1, seq, 1).expand(B, nk, seq, Dh)
    past = KVPast(keys.clone(), keys.clone())
    cache = H2OKVCache(
        hh_size=hh_sz,
        recent_size=rz,
        layer_idx=0,
        num_attention_heads=8,
        num_key_value_heads=nk,
        num_key_value_groups=8,
    )
    cache.position_ids = torch.arange(seq).view(1, 1, -1).expand(B, nk, -1).long()
    s = torch.zeros(B, nk, seq)
    s[..., :] = torch.arange(seq, dtype=torch.float).reshape(1, nk, seq)
    s[..., :6] -= 4444
    s[..., 7] += 9900
    s[..., 8] += 9800
    s[..., 9] += 9777
    cache.hh_score = s
    out = cache.evict_for_space(past, num_coming=2048, padding_attention_mask=attn)[0]
    kept = sorted(out[0, 0, :, 0].round().long().tolist())
    assert kept[-rz:] == list(range(pref, seq))
    hh_set = set(kept[:hh_sz])
    assert hh_set.issubset({7, 8, 9}) and hh_set.issuperset({9})


def _evict_budget_same_setup_batched(
    attn: torch.Tensor,
    nk: int,
    seq: int,
    Dh: int,
    hh_sz: int,
    rz: int,
    num_heads: int = 8,
    num_kv_groups: int = 8,
):
    """Build a deterministic batched eviction scenario reused by padded-budget tests.

    ``attn``: ``(B, seq)`` bool row masks. Keys store ``float(column_index)`` per slot so post-eviction gathers
    can be decoded as integers; ``hh_score`` starts zero unless caller replaces it.

    Returns ``(H2OKVCache, KVPast)`` with canonical ``position_ids = arange(seq)`` tiled per KV head.
    """

    B = attn.shape[0]
    keys = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1).expand(B, nk, seq, Dh)
    past = KVPast(keys.clone(), keys.clone())
    cache = H2OKVCache(
        hh_size=hh_sz,
        recent_size=rz,
        layer_idx=0,
        num_attention_heads=num_heads,
        num_key_value_heads=nk,
        num_key_value_groups=num_kv_groups,
    )
    pid = torch.arange(seq, dtype=torch.long).view(1, 1, -1).expand(B, nk, -1)
    cache.position_ids = pid.clone()
    cache.hh_score = torch.zeros(B, nk, seq)
    return cache, past


def test_evict_budget_uses_valid_counts_not_physical_seq_batch_two_rows():
    """Contrast padding-aware eviction budget vs naive physical length across two padded rows.

    Both rows share ``seq_len=14`` exceeding ``hh+recent``, but only seven KV columns are “real” per mask so
    ``valid_counts + num_coming <= cache_size`` skips eviction when ``padding_attention_mask`` is passed.
    Same tensors without mask trim to capacity; assertions use ``evict_for_space(...)[0]`` return values since
    ``KVPast`` is not always mutated in place.
    """

    B, nk, seq, Dh = 2, 1, 14, 2
    hh_sz, rz = 4, 3
    cache_capacity = hh_sz + rz

    attn = torch.zeros(B, seq, dtype=torch.bool)
    for b in range(B):
        attn[b, seq - 7 :] = True  # 7 valid HF-style column ids per row (matching padding row width)

    vkm = build_h2o_kv_valid_mask(
        kv_position_ids=torch.arange(seq).view(1, 1, -1).expand(B, nk, -1),
        attention_mask=attn,
    )
    assert vkm is not None
    assert (vkm.sum(dim=-1) == 7).all()

    cache, past = _evict_budget_same_setup_batched(attn, nk, seq, Dh, hh_sz, rz)

    seq_before = past[0].shape[2]
    out_kept = cache.evict_for_space(past, num_coming=0, padding_attention_mask=attn)[0]

    assert out_kept.shape[2] == seq_before == seq

    naive_cache, naive_past = _evict_budget_same_setup_batched(attn, nk, seq, Dh, hh_sz, rz)
    naive_k = naive_cache.evict_for_space(naive_past, num_coming=0, padding_attention_mask=None)[0]
    assert naive_k.shape[2] == cache_capacity


def test_evict_batch_masked_padding_never_kept_as_heavy_hitters():
    """Regress masking when pad slots would dominate greedy top-k within the hh candidate slice.

    ``B=2`` with asymmetric left-padding: assign near-maxfloat scores on every pad KV index inside the hh window
    but tiny positives on reals; after eviction each row’s hh prefix must encode only masked-``True`` column ids
    and trailing ``recent_size`` retains the lexical tail ``seq-recent .. seq-1``.
    """

    B, nk, seq, Dh = 2, 1, 30, 2
    hh_sz, rz = 5, 10  # hh_cand_hi = seq - rz = 20
    attn = torch.zeros(B, seq, dtype=torch.bool)
    attn.fill_(False)
    attn[0, 15:] = True  # reals: slot ids 15..29 (left-padded HF-style)
    attn[1, 10:] = True  # reals: 10..29
    hh_cand_hi = seq - rz

    cache, past = _evict_budget_same_setup_batched(attn, nk, seq, Dh, hh_sz, rz)
    big = torch.finfo(torch.float32).max / 4096

    score = torch.zeros(B, nk, seq)
    for b in range(B):
        for t in range(hh_cand_hi):
            if not bool(attn[b, t]):
                score[b, 0, t] = big * (hh_cand_hi - t + 1)
            else:
                score[b, 0, t] = float(t) * 0.003 + float(b)

    torch.manual_seed(2)
    score[..., :hh_cand_hi] = score[..., :hh_cand_hi] + torch.randn(B, nk, hh_cand_hi) * 1e-4
    cache.hh_score = score

    assert (seq + 0) > (hh_sz + rz), "setup should exceed naive cache budget"

    out_k = cache.evict_for_space(past, num_coming=10_000, padding_attention_mask=attn)[0]
    capacity = hh_sz + rz

    want_recent = tuple(range(seq - rz, seq))
    for b in range(B):
        row = out_k[b, 0].clone()
        for k in range(hh_sz):
            slot_idx = int(round(float(row[k, 0])))
            assert bool(attn[b, slot_idx]), f"H2O kept padding slot idx={slot_idx} for batch={b}"
        recent_part = tuple(int(round(float(row[k, 0]))) for k in range(hh_sz, capacity))
        assert recent_part == want_recent


def test_attn_scaling_matches_head_dim_inverse_sqrt(qwen_cfg):
    """Module wiring check: HF-derived ``head_dim`` should drive attention ``scaling`` (= ``head_dim**-0.5``).

    Uses ``text_config`` when multimodal checkpoints wrap a nested text config clone.
    """

    txt = getattr(qwen_cfg, "text_config", None) or qwen_cfg
    cfg = copy.deepcopy(txt)
    att = H2OQwen3_5Attention(cfg, layer_idx=0)
    assert math.isclose(att.scaling, att.head_dim**-0.5)


def test_eager_attn_matches_scaled_softmax(qwen_cfg):
    """Numerical parity: transformers ``eager_attention_forward`` on ``H2OQwen3_5Attention`` vs manual softmax.

    Supplies KV tensors in **compact** KV-head layout plus ``repeat_kv`` on the manual path—mirrors HF usage and
    tolerates bf16 internals via FP32 softmax reference (threshold ``3e-2``).
    """

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        eager_attention_forward,
        repeat_kv,
    )

    txt = getattr(qwen_cfg, "text_config", None) or qwen_cfg
    cfg = copy.deepcopy(txt)
    attn = H2OQwen3_5Attention(cfg, layer_idx=0).float().eval()
    torch.manual_seed(11)
    for p in attn.parameters():
        torch.nn.init.normal_(p, 0.02)

    bs, nt = 1, cfg.num_attention_heads
    hd = attn.head_dim
    nl = 5
    q = torch.randn(bs, nt, nl, hd, dtype=torch.float32)
    k = torch.randn(bs, cfg.num_key_value_heads, nl, hd, dtype=torch.float32)
    v = torch.randn(bs, cfg.num_key_value_heads, nl, hd, dtype=torch.float32)
    addm = torch.zeros(bs, nt, nl, nl)
    yo, _ = eager_attention_forward(attn, q, k, v, addm, attn.scaling, dropout=0.0)
    kxp = repeat_kv(k, attn.num_key_value_groups)
    vxp = repeat_kv(v, attn.num_key_value_groups)
    sc = torch.matmul(q, kxp.transpose(-2, -1)).float().mul(attn.scaling)
    pr = torch.softmax(sc + addm, dim=-1, dtype=torch.float32).to(q.dtype)
    yref = torch.matmul(pr, vxp).transpose(1, 2).contiguous()
    assert torch.max(torch.abs(yo.float() - yref.float())).item() < 3e-2


def _patch_kv_budget(cfg, hh: int, recent: int):
    """Clone ``cfg`` then set ``hh_size`` / ``recent_size`` on both top-level objects and nested ``text_config``.

    Mirrors how ``modify_qwen`` reads ``getattr(..., hh_size/recent_size)`` for ``H2OKVCache``.
    """

    c = copy.deepcopy(cfg)
    for obj in filter(None, (c, getattr(c, "text_config", None))):
        setattr(obj, "hh_size", hh)
        setattr(obj, "recent_size", recent)
    return c


def _first_h2o_self_attn_layer(model):
    """Scan decoder ``language_model.layers`` until first standard (non-linear) attention replaced by ``H2OQwen3_5Attention``.

    Used by GPU bookkeeping tests that introspect ``layer_idx`` aligned with DynamicCache buckets.
    """

    lm = model.model.language_model
    for lyr in lm.layers:
        sa = getattr(lyr, "self_attn", None)
        if sa is not None and sa.__class__.__name__ == "H2OQwen3_5Attention":
            return lyr
    raise AssertionError("no H2OQwen3_5Attention layer found")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gpu_generate_and_reset_cache(qwen_cfg, batch_size):
    """End-to-end smoke: load quantized bf16/fp16 H2O Causal LM on CUDA, ``generate`` a few greedy tokens.

    Covers single-string vs batched tokenizer output; post-run ``reset_h2o_state`` requires every replaced layer’s
    ``kv_cache`` to drop ``hh_score`` and ``position_ids`` (generation cache teardown contract).
    """

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_kv_budget(qwen_cfg, hh=8, recent=8)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = H2OQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    if batch_size == 1:
        batch = tok("5+7=", return_tensors="pt").to("cuda")
    else:
        batch = tok(
            ["9+1=", "short question one word sky color"],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    torch.cuda.empty_cache()
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
    assert out.shape[0] == batch_size
    assert torch.isfinite(out.float()).all()

    model.reset_h2o_state()
    for lyr in model.model.language_model.layers:
        sa = getattr(lyr, "self_attn", None)
        if sa is None or sa.__class__.__name__ != "H2OQwen3_5Attention":
            continue
        assert sa.kv_cache.hh_score is None and sa.kv_cache.position_ids is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_long_prefill_past_within_cache_cap(qwen_cfg):
    """Prefetch long prompt (~220 counted tokens): ensure DynamicCache KV depth never exceeds configured H2O cap.

    Validates each ``H2OQwen3_5Attention`` layer’s persisted keys stay ``<= hh_size + recent_size`` after a single forward.
    """

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_kv_budget(qwen_cfg, hh=12, recent=12)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = H2OQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    long_text = "counting " + " ".join(str(i) for i in range(220))
    batch = tok(long_text, return_tensors="pt").to("cuda")
    model.reset_h2o_state()
    with torch.inference_mode():
        out = model(**batch, use_cache=True, return_dict=True)
    pst = out.past_key_values
    assert pst is not None
    lm = model.model.language_model
    for li, lyr in enumerate(lm.layers):
        if (
            not hasattr(lyr, "self_attn")
            or lyr.self_attn.__class__.__name__ != "H2OQwen3_5Attention"
        ):
            continue
        cap = lyr.self_attn.kv_cache.cache_size
        layer = pst.layers[li]
        if layer.keys is None:
            continue
        assert layer.keys.shape[2] <= cap


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_greedy_padded_decode_position_ids_stable(qwen_cfg):
    """Stateful decode audit on a heterogeneous-length batch (large ``hh/recent`` to avoid eviction churn).

    After padded prefill and five greedy one-token forwards, verifies per step: KV length aligns with DynamicCache depth,
    ``h2o_next_position`` matches ``max(kv_cache.position_ids[..., -1]) + 1``, and padded ``attention_mask`` columns
    are never marked valid via redundant gather vs ``build_h2o_kv_valid_mask``.
    """

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    tok.padding_side = "left"
    cfg = _patch_kv_budget(qwen_cfg, hh=96, recent=96)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = H2OQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        config=cfg,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    prompts = ["9+1=", "short question sky color yes"]
    batch = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    ref_lyr = _first_h2o_self_attn_layer(model)
    ref_sa = ref_lyr.self_attn
    li = ref_sa.layer_idx

    def verify(past, attn_cuda: torch.Tensor, tag: str):
        """Internal helper reused after each forward: asserts ``past`` coherence for layer ``li`` tracked above.

        Compares CUDA ``past_key_values`` to CPU-gather copies of tokenizer mask; detects ``vm & in_bounds & ~gathered``
        (valid mask claims for true pad indices) separately from legitimate left-padding KV slots.
        """

        pst = past
        am_bool = attn_cuda.detach().cpu().bool()
        pst_len = pst.get_seq_length()
        lay = pst.layers[li]
        assert lay.keys is not None
        kv_len = lay.keys.shape[2]
        pid = ref_sa.kv_cache.position_ids
        assert pid is not None and pid.shape[-1] == kv_len
        assert kv_len == pst_len, tag

        h2 = pst.h2o_next_position
        assert h2 is not None and h2.ndim == 1 and tuple(h2.shape) == tuple(am_bool.shape[:1])
        pred_next = pid[:, :, -1].max(dim=1).values.to(dtype=torch.long) + 1
        assert torch.equal(
            h2.detach().cpu(), pred_next.detach().cpu()
        ), f"{tag} h2o_next={h2} pred={pred_next}"

        pid_cpu = pid.detach().cpu()
        vm = build_h2o_kv_valid_mask(pid_cpu, am_bool)
        assert vm is not None
        in_bounds = pid_cpu < am_bool.shape[-1]
        safe_idx = pid_cpu.clamp(min=0, max=am_bool.shape[-1] - 1).long()
        expanded = am_bool[:, None, :].expand(-1, pid_cpu.shape[1], -1)
        gathered = expanded.gather(dim=-1, index=safe_idx)
        bogus_valid = vm & in_bounds & (~gathered)
        assert not bogus_valid.any(), f"{tag} padded column marked valid KV"

    model.reset_h2o_state()

    inp_ids = batch["input_ids"]
    attn = batch["attention_mask"]

    with torch.inference_mode():
        out = model(
            input_ids=inp_ids,
            attention_mask=attn,
            use_cache=True,
            return_dict=True,
        )
        pst = out.past_key_values
        verify(pst, attn, "prefill")

        for step in range(5):
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            attn = torch.cat([attn, torch.ones_like(nxt, dtype=attn.dtype)], dim=-1)
            out = model(
                input_ids=nxt,
                attention_mask=attn,
                past_key_values=pst,
                use_cache=True,
                return_dict=True,
            )
            pst = out.past_key_values
            verify(pst, attn, f"decode_{step}")

