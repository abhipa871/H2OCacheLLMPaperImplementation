# H2O-style KV cache (Qwen3.5)

This repo implements **H2O-style KV cache eviction** inside Hugging Face **dense Qwen3.5** models: `H2OKVCache` plus `H2OQwen3_5ForCausalLM` in [`modify_qwen.py`](modify_qwen.py). It is **not** a generic drop-in for other architectures.

The idea follows the H2O heavy-hitter + recent-window cache policy for long-context LLM inference; this code paths it through Qwen3.5’s attention stack and `DynamicCache`.

## Layout

- [`modify_qwen.py`](modify_qwen.py) — `H2OKVCache`, `H2OQwen3_5Attention` / `H2OQwen3_5ForCausalLM`, and mask helpers (`build_h2o_kv_valid_mask`, `make_h2o_causal_mask`).
- [`test_modify_qwen_h2o.py`](test_modify_qwen_h2o.py) — CPU coverage plus optional CUDA `generate` / cache-cap checks.
- [`h2o_gsm8k_eval_colab.ipynb`](h2o_gsm8k_eval_colab.ipynb) — GSM8K accuracy vs KV **budget %** (Local / Full / H2O sweeps; Colab-oriented).

## Requirements

- Python 3, **PyTorch**, **Transformers** (notebook uses `transformers>=4.53.0`; use a matching or newer version locally).
- **CUDA** optional; GPU-only tests are marked `gpu`.
- For tests: `pytest`.

```bash
pip install -U "transformers>=4.53.0" torch datasets accelerate matplotlib pandas tqdm pytest
```

## Using `H2OQwen3_5ForCausalLM` in Python

`modify_qwen.py` reads **`hh_size`** and **`recent_size`** from the model config (via each attention layer’s `H2OKVCache`). Effective cap is **`hh_size + recent_size`** tokens in the H2O cache.

Set those fields on **both** the top-level `config` and **`config.text_config`** when it exists (same pattern as `_patch_kv_budget` in the tests). For **recent-only “Local”** style behavior (no heavy hitters), use **`hh_size=0`** and set **`recent_size`** to your recent window in tokens.

Use **`attn_implementation="eager"`** with this stack. For **batched** generation, set **`tokenizer.padding_side = "left"`** (see `H2OKVCache` docstring). Call **`model.reset_h2o_state()`** when you want to clear H2O score/position bookkeeping between unrelated sequences (after `generate` in the tests).

```python
import copy
import torch
from transformers import AutoConfig, AutoTokenizer

from modify_qwen import H2OQwen3_5ForCausalLM


def patch_h2o_budget(cfg, hh_size: int, recent_size: int):
    """Mirror test_modify_qwen_h2o._patch_kv_budget: set on root + text_config."""
    c = copy.deepcopy(cfg)
    for obj in (c, getattr(c, "text_config", None)):
        if obj is None:
            continue
        setattr(obj, "hh_size", hh_size)
        setattr(obj, "recent_size", recent_size)
        # Absolute (integer budget) runs: leave ratio bookkeeping unset unless you rely on notebook helpers below.
        for k in ("hh_ratio", "h2o_full_cache_size"):
            if hasattr(obj, k):
                setattr(obj, k, None)
    return c


MODEL_ID = "Qwen/Qwen3.5-0.8B"

base_cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
# Example budgets: hh “heavy hitter” slots + trailing “recent” window (tokens).
cfg = patch_h2o_budget(base_cfg, hh_size=8, recent_size=8)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device_kw = {} if torch.cuda.is_available() else {"device_map": "cpu"}
if torch.cuda.is_available():
    device_kw["device_map"] = "cuda:0"

model = (
    H2OQwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        **device_kw,
    )
    .eval()
)

batch = tokenizer("5+7=?", return_tensors="pt")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = {k: v.to(dev) for k, v in batch.items()}

with torch.inference_mode():
    ids = model.generate(
        **batch,
        max_new_tokens=24,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(ids[0], skip_special_tokens=True))
model.reset_h2o_state()
```

### Ratio-style settings in the GSM8K notebook

The Colab notebook also defines **`load_h2o_model_ratio` / `load_h2o_model_absolute`**. **`load_h2o_model_absolute`** matches the integer `hh_size` / `recent_size` knobs above after clearing optional ratio metadata. **`load_h2o_model_ratio`** plus **`budget_to_kv_settings`** encode how Full/H2O curves share the KV budget in that experiment pipeline. If you copy logic into your own scripts, ensure the **integer** `hh_size` / `recent_size` reflect the budgets you intend (same pattern as **`patch_h2o_budget`** here).

## GSM8K notebook (Colab / local)

1. Use a **GPU** runtime if possible.
2. Run the install cell, e.g.  
   `pip install -U "transformers>=4.53.0" datasets accelerate matplotlib pandas tqdm`
3. Make [`modify_qwen.py`](modify_qwen.py) importable (repository root on `PYTHONPATH`, or upload next to the notebook in Colab as in its instructions).
4. Run setup + evaluation sections in order.

Typical artifacts: **`results.csv`**, **`predictions.csv`**, **`results.json`**, **`accuracy_vs_budget.png`**.

## Tests

From the repo root:

```bash
# CPU-centric tests
python -m pytest test_modify_qwen_h2o.py -m "not gpu"

# CUDA smoke tests (needs a GPU)
python -m pytest test_modify_qwen_h2o.py -m gpu
```

## Troubleshooting

- **`unexpected keyword argument 'padding_attention_mask'`** — Qwen decoder layer forward must thread custom kwargs through to `self_attn` (notebook “Sanity checks” cell expands on this).
- **KV length / `position_ids` mismatches** — compare `kv_cache.position_ids` to `key_states.shape[2]` and the logic in `_build_additive_attention_mask` in [`modify_qwen.py`](modify_qwen.py).
- **Heavy-hitter masking** — very small valid regions vs **`hh_size`** can make top-k behavior noisy; warnings may come from `_mask_hh_scores_with_valid_positions`.

## Reference

Implements eviction behavior in the spirit of **H2O** (heavy-hitter + recent KV retention) for efficient long-context inference; see the H2O paper for the original policy.
