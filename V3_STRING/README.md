# V-JEPA 3 — reading guide

> **What this is.** `app/vjepa_3/` is a fork of `app/vjepa_2_1/` that makes two
> positional-encoding choices configurable instead of hardcoded. Every feature
> is behind a flag whose default reproduces V-JEPA 2.1 **bitwise**, so 2.1 is a
> true A/B control arm.
>
> Companion: [`../V2_1_ANALYSIS/README.md`](../V2_1_ANALYSIS/README.md) — what
> V-JEPA 2.1 itself does. Read that first if you haven't.
>
> Status: **Stages 0–4 built and gated.** 65 tests pass. Deferred work in §7.

---

## 1. Why this fork exists

V-JEPA 2.1's RoPE makes two arbitrary choices:

**(a) A hard three-way partition of every attention head.**
`RoPEAttention` splits `head_dim=64` into `d_dim = h_dim = w_dim = 20`
([modules.py:253-255](../app/vjepa_3/models/utils/modules.py)), permanently
assigning dims 0–19 to time, 20–39 to height, 40–59 to width, 4 unrotated.
Nothing says the axes shouldn't share subspaces. **Cayley-STRING** replaces
this with a learned orthogonal basis (§4).

**(b) RoPE base 10000, inherited from language models.** LLMs have thousands of
positions; V-JEPA has 8–32. Most of the spectrum never turns. Measured sweep
in radians across the *entire* grid extent, `dim=20` (10 frequencies):

| axis / grid | sweep per channel | channels > 0.2 rad |
|---|---|---|
| temporal @16f (8 pos) | 7.00 2.79 1.11 .44 .18 .07 .03 .011 .004 .002 | **4 / 10** |
| temporal @64f (32 pos) | 31.0 12.3 4.91 1.96 .78 .31 .12 .05 .02 .008 | 6 / 10 |
| spatial @256px (16 pos) | 15.0 5.97 2.38 .95 .38 .15 .06 .02 .009 .004 | 5 / 10 |

Reproduce:
```bash
python3 -c "n=10; om=[10000**(-i/n) for i in range(n)]; print([round(o*7,3) for o in om])"
```

Lowering the base compresses the spectrum onto the range that actually exists:
θ=1000 → 6/10 live, θ=100 → 8/10. The STRING paper independently initialises
with max wavelength 100 (following RoPE-Mixed) — convergent evidence. Note even
θ=100 does not revive all ten; it widens the band, it doesn't saturate it.

---

## 2. How to read this code

**Read it as a diff against 2.1, not as a standalone tree.** 9 of 11 files are
byte-identical copies; all the thinking lives in one file.

```bash
diff -r app/vjepa_2_1 app/vjepa_3 -x '__pycache__'
```

### Reading order

| # | Read | Why |
|---|---|---|
| 1 | `diff` above | Establishes how small the change surface is |
| 2 | [`modules.py:17-160`](../app/vjepa_3/models/utils/modules.py) — `DEFAULT_ROPE_THETA`, `build_omega`, `_normalize_rope_theta`, `rotate_queries_or_keys` | The whole frequency story. `build_omega`'s docstring carries the dead-channel table. |
| 3 | `modules.py:270-300` — the `omega_*` buffer block in `RoPEAttention.__init__` | The one genuinely subtle decision in the fork (§4) |
| 4 | `modules.py:358-365` — adjustable temporal spacing | 3 lines, VideoRoPE's ATS |
| 5 | [`string.py`](../app/vjepa_3/models/utils/string.py) — module docstring first | The `Pᵀ` cancellation derivation, then ~40 lines of code |
| 6 | `modules.py` — the `string_P` block in `RoPEAttention.forward` | Where STRING actually acts: 4 lines |
| 7 | [`test_vjepa3_rope.py`](../tests/models/test_vjepa3_rope.py), [`test_vjepa3_string.py`](../tests/models/test_vjepa3_string.py) | The tests *are* the spec — each pins a claim made above |
| 8 | `train.py:110-121` → `utils.py:214` → `vision_transformer.py:61` → `modules.py` | The config hop chain (§5) |

### The general technique for this codebase

1. **Find the call site before reading the definition.** A helper's shapes are
   opaque alone and obvious once you see one caller
   (`grep -rn "function_name" app/ src/`).
2. **Annotate shapes as you go** — `# (B, N, D)`. If you can't write the shape,
   you don't understand the line yet.
3. **Separate "always true" from "this config's choice."** Branches gated on a
   flag are optional behaviour; understand the default path first.

---

## 3. File map

| File | Status |
|---|---|
| `models/utils/string.py` | **NEW** — `CayleySTRING`, `apply_string`, and the derivation |
| `models/utils/modules.py` | **MODIFIED** — frequency control + the `string_P` application point |
| `models/vision_transformer.py` | owns the `CayleySTRING`; computes bases once per forward, threads `P[i]` into each block |
| `models/predictor.py` | same, behind its own `use_string_predictor` flag |
| `train.py` | 6 new `cfgs_model.get` + `EMBED_DIM_BY_MODEL_NAME` (§6) |
| `utils.py` | 6 new `init_video_model` params; `init_opt` weight-decay exclusion (§4.2) |
| `wrappers.py`, `transforms.py`, `patch_embed.py`, `pos_embs.py`, `masks_dist.py` | unchanged copies |
| `models/utils/embeddings.py` | **DEAD** — nothing imports it. Removed from 2.1 after the fork snapshot; retained here only as an artifact. Safe to delete. |

**Frozen — never edit:** `app/vjepa_2_1/` and `src/`. `src/hub/backbones.py:227`
imports `app.vjepa_2_1.models` and loads `strict=True`; the ek100 eval imports it
too. More importantly, an untouched 2.1 is what makes the A/B trustworthy.

---

## 4. The two subtle decisions

### 4.1 Why `omega_*` is `None` at the default base

The obvious implementation precomputes each axis's frequency table as a buffer.
**That silently breaks bitwise parity.** Building in fp32 and casting to bf16 is
not equal to building in bf16 throughout — `omega /= n` and `1/θ**omega` round
differently:

```
fp32→bf16 : ... 0.025146484375 ... 0.00396728515625  0.0015869140625  ...
native bf16: ... 0.0250244140625 ... 0.003936767578125 0.0016021728515625 ...
```

4 of 10 channels diverge at `head_dim=64`. Under `autocast(bfloat16)` a
precomputed buffer would perturb numerics *even at θ=10000*.

So: **an axis left at the default base holds `None`** and takes the legacy
inline path inside `rotate_queries_or_keys`, which is byte-identical to 2.1.
Only an axis the config actually changes materialises a table. Buffers are
`persistent=False`, so they never enter the `state_dict` and can't break a
strict load.

Pinned by `test_fp32_precompute_does_NOT_match_legacy_bf16`. That test asserts
inequality on purpose — if it ever starts passing, someone changed the dtype
handling and parity is gone.

### 4.2 Cayley-STRING, and the DDP rule that is *not* what you'd assume

`STRING(q, r) = Pᵀ D(r) P q`. The outer `Pᵀ` cancels inside the inner product:

```
<Pᵀ D(r) P q , Pᵀ D(r') P k> = qᵀ Pᵀ D(r)ᵀ D(r') P k = qᵀ Pᵀ D(r'−r) P k
```

so it suffices to left-multiply `q` and `k` by `P` before the existing rotation.
`P = (I−S)(I+S)⁻¹` with `S = W−Wᵀ`; antisymmetric `S` has purely imaginary
eigenvalues, so `(I+S)` is always invertible and `P` is always orthogonal.
`W = 0 ⇒ P = I`, which is the warm-start story: load a 2.1 checkpoint into a
`use_string: true` model, `string.weight` is the only missing key, and training
starts **exactly** at the 2.1 solution.

Invariance survives the block-diagonal `D(r) = blockdiag(D_d, D_h, D_w, I₄)`
because each block satisfies `D(p)ᵀD(p') = D(p'−p)` and the 4 unrotated dims are
the identity = rotation by zero. `P` is full `head_dim × head_dim`, so it mixes
**across** the d|h|w partition — that mixing is the point.

> **Hard dependency.** This needs the *correct* rotary construction
> (`repeat_interleave`). `src/models/utils/modules.py:42-43` uses `.repeat(...)`
> (tiling), which pairs adjacent channels with different frequencies — not a
> rotation. **Verified**: substituting the tiled version makes
> `TestTranslationInvariance` fail, with and without STRING.

**The DDP rule, measured rather than assumed** (torch 2.13). The encoder is
wrapped `DistributedDataParallel(..., static_graph=True)`:

| pattern | result |
|---|---|
| parameter consistently unused on a rank, same every iteration | **accepted** — static_graph supports unused params when the set is constant. But that rank still joins the allreduce contributing nothing, so the averaged gradient is scaled down: a silent effective-LR change, not a crash. |
| usage varying **across iterations** on one rank | `RuntimeError: ...your training graph has changed...` plus a hang on its peer |

So the rule is *"the used-parameter set must be constant across iterations on
each rank"*, **not** the stronger "must be used on every step". Since
`img_rank_ratio` pins each rank to one modality for the whole run, a
modality-gated `P` would land in the first row — degrading quietly rather than
failing loudly, which is worse. V-JEPA 3 computes `P` unconditionally and
sidesteps both. Both rows are pinned, with a negative control, by
`tests/distributed/test_vjepa3_ddp_static_graph.py`.

### 4.3 Weight decay would unlearn STRING

`init_opt` grouped parameters by `("bias" not in n) and (len(p.shape) != 1)`.
A `[L, D, D]` STRING weight is ndim 3 with no "bias" in the name, so it landed
in the **decay** group — and decay pulls `W → 0`, hence `P → I`, unlearning the
basis. `app/vjepa_3/utils.py:init_opt` now uses explicit predicates, gives
STRING its own `WD_exclude: True, weight_decay: 0` groups, builds **lists**
rather than the generators 2.1 used (generators are consumed on first
iteration, so they cannot be asserted over), and asserts the groups *partition*
the parameters.

---

## 5. Config reference

```yaml
model:
  rope_theta: 10000.0                # scalar, or [d, h, w] with h == w
  rope_temporal_scale: 1.0           # VideoRoPE adjustable temporal spacing
  use_string: false                  # Cayley-STRING on the encoder
  use_string_predictor: false        # ...and on the predictor, separately
  string_share_across_layers: false  # true -> one [1,D,D] for the whole stack
  string_init_std: 0.0               # 0.0 => P = I => exact 2.1 baseline
```

`use_string` / `use_string_predictor` are separate flags because the
predictor's RoPE never receives `T`/`H_patches`/`W_patches` and runs on
re-sorted token sequences — a genuinely different positional regime.

`string_share_across_layers: false` costs `depth × 64 × 64` params
(98K for ViT-L, ~0.03% of the model).

`rope_theta` — RoPE base / max wavelength. `theta_h != theta_w` **raises**:
unequal spatial bases mean a 90° rotation of the input no longer corresponds to
relabelling the encoding, so the model acquires a preferred image axis. The
temporal base is free to differ — that asymmetry is the point.

`rope_temporal_scale` — scales the temporal index. `< 1.0` compresses; the
cheapest mitigation for the 16f→64f shift between pretrain and cooldown
(8 vs 32 temporal positions through one frequency table).

Ablation arms: `10000` (control) · `1000` · `100` (paper) ·
`[100, 10000, 10000]` (temporal-only rebase, isolates the dead-channel finding).

### The hop chain — where keys silently vanish

A new `model:` key does **nothing** until read explicitly. `VisionTransformer`,
`VisionTransformerPredictor` and `Block` all end in `**kwargs`, which swallows
unknown names with no error. Three live instances of this exist in the repo
today: `grid_depth` passed to `Block` (no such parameter), `use_SiLU`/`wide_SiLU`
miscased at `src/hub/backbones.py:232`, and `local_window`/`vit_conv` in the
yaml with no reader anywhere.

```
configs/train_3/*/*.yaml   model: rope_theta
  -> train.py:113          rope_theta = cfgs_model.get("rope_theta", 10000.0)
  -> train.py:390          init_video_model(..., rope_theta=rope_theta)
  -> utils.py:214          def init_video_model(..., rope_theta=10000.0)
  -> utils.py:235 / :266   encoder AND predictor factory calls
  -> vision_transformer.py:60 / predictor.py:59   NAMED param, not **kwargs
  -> Block(...)  :142 / :169
  -> modules.py:509        Block.__init__ NAMED param
  -> modules.py:270        RoPEAttention.__init__ -> self.omega_*
```

`test_theta_and_temporal_scale_change_the_output` guards the whole chain: if any
hop is missed, the knob stops moving the output and the test fails.

---

## 6. Pre-existing 2.1 defects found while forking

| Defect | Status in v3 |
|---|---|
| `embed_dim_encoder` unbound for any model outside {`vit_large`, `vit_giant_xformers`, `vit_gigantic_xformers`} — the `else` branch only `print`ed, so `configs/train_2_1/vitb16` raises `NameError`. | **FIXED** — `EMBED_DIM_BY_MODEL_NAME` covers all 14 factories, unknown names raise. A test cross-checks the table against the factories. |
| `use_rope=False` reads `self.pos_embed`, which the 2.1 ViT never assigns → `AttributeError`. The entire additive sin/cos path (`pos_embs.py`) is unreachable dead code. | **DOCUMENTED, not fixed** — pinned by a test asserting *both* trees raise, so the control arm stays a true control. All shipped configs use `use_rope: true`. |
| `src/models/utils/modules.py:42-43` uses `.repeat(1,1,1,2)` (tiling) where the 2.1 fork uses `repeat_interleave` — tiling pairs adjacent channels with *different* frequencies, so it is not a rotation. Acknowledged in-source; kept for V-JEPA 2 checkpoint compat. | **AVOIDED** — v3 inherits the fork's correct version. Matters for Stage 3: STRING's translation invariance requires `D(r)ᵀD(r') = D(r'−r)`, which tiling breaks. |
| `all_hierarchical_layers` unbound for any `pred_depth` outside {4, 8, 12, 20, 24, 40} — no `else`, so it failed on the next line with a bare `UnboundLocalError` naming the variable rather than the unsupported depth. | **FIXED** — table lookup plus a `ValueError` that names `pred_depth`. Same defect class as `embed_dim_encoder`. |
| `from src.utils.adamw import AdamW` — that module does not exist. Fires only when `use_radamw: true`. | Carried over, still dead. |
| `interpolate_rope` divides by `(H_patches - 1)` → ZeroDivisionError on a 1-patch input. | Carried over, out of scope. |

---

## 7. Not yet built

- **VideoRoPE LTA** — deferred, and likely *harmful* here: it pushes temporal
  toward the slowest frequencies, i.e. into the band that is already dead at 8
  positions (§1). LTA is calibrated for VLMs with hundreds of frames. Revisit
  only above ~64 frames, and preferably together with `rope_theta < 10000`.
- **VideoRoPE Diagonal Layout** — out of scope. It reconciles text-token and
  video-token indices in a shared VLM position space; V-JEPA has no text tokens
  and no shared index space. Recorded so it is not re-litigated.
- **qkv folding** — `P·W_q` can be absorbed into the projection weight,
  recovering STRING's ~3–4% encoder overhead. Only worth doing once STRING has
  trained stably; the unfolded form stays the reference implementation.
- **Eval plumbing** — 6 of 7 `configs/eval_2_1` benchmarks route through
  `src/models/vision_transformer` and its tiled RoPE, so evaluating a v3
  checkpoint through them would compare STRING against a *different* positional
  encoding. The fix is purely additive: new `modelcustom/*_v3.py` importing
  `app.vjepa_3.models`, new `configs/eval_3/`, and a `use_v3` branch in
  `vit_encoder_predictor_concat_ar.py:37-42`. **Do this before trusting any
  ablation number.**
- **Ablation config matrix** — one config per arm (§5), plus `ABLATIONS.md`.

---

## 8. Verify

```bash
python3 -m pytest tests/models/ tests/distributed/ -v
```

65 pass, 3 skip (CUDA-gated). CPU only, ~20s.

`vit_tiny` is `embed_dim=192 / num_heads=3 ⇒ head_dim=64` — the **same**
`head_dim` as ViT-L (1024/16) and ViT-g (1408/22), so `d_dim=h_dim=w_dim=20`
with 4 unrotated dims. The tiny model reproduces production rotary geometry
exactly. Do not widen it.

| Test file | Pins |
|---|---|
| `test_vjepa3_parity.py` | v3 ≡ 2.1 bitwise (video / image / masked), identical state_dict, the `EMBED_DIM` table, the `use_rope=False` breakage |
| `test_vjepa3_rope.py` | `build_omega` ≡ legacy bitwise across dtypes, the dead-channel counts, the fp32/bf16 negative result, isotropy validation, the hop chain |
| `test_vjepa3_string.py` | `P = I` at zero-init, orthogonality and `det = 1`, translation invariance, cls/register scores unchanged, antisymmetric gradient, activation-checkpointing agreement, predictor wiring, weight-decay exclusion |
| `test_vjepa3_configs.py` | every `model:` key has a reader in train.py — the check that catches `local_window`-style orphans |
| `test_vjepa3_ddp_static_graph.py` | STRING trains across a T=1 rank and a T=8 rank under `static_graph=True`, **with a negative control that must fail** |

### On negative controls

Two of these tests assert that something *fails*, and both earned their place:

* `test_fp32_precompute_does_NOT_match_legacy_bf16` caught a design in the
  original plan (precomputed `omega` buffers) that would have silently broken
  parity.
* The DDP negative control **initially passed**, which meant it was proving
  nothing. Investigating why produced the corrected rule in §4.2 — the
  first version gated on `T == 1`, a pattern static_graph happily accepts.

A test that cannot fail is not evidence. When adding one here, verify it fails
when it should, and *for the right reason*.
