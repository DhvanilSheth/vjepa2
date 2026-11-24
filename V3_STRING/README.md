# V-JEPA 3 — reading guide

> **What this is.** `app/vjepa_3/` is a fork of `app/vjepa_2_1/` that turns
> hardcoded architecture choices into config flags: two positional-encoding
> choices (§1–§5), and the feed-forward and normalization layers (§9). Every
> feature is behind a flag whose default reproduces V-JEPA 2.1 **bitwise**, so
> 2.1 is a true A/B control arm. §9 also covers the isotropy diagnostic, the
> clustering probe and the single-GPU ablation ladder built on those flags.
>
> Companion: [`../V2_1_ANALYSIS/README.md`](../V2_1_ANALYSIS/README.md) — what
> V-JEPA 2.1 itself does. Read that first if you haven't.
>
> Status: **Stages 0–10 built and gated.** 141 tests pass. `train.py` has
> still never been run end to end; that and the other gaps are in §7.

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

**Read it as a diff against 2.1, not as a standalone tree.** Most files are
byte-identical copies. The positional work lives almost entirely in
`modules.py` and `string.py`; the §9 additions are each one small new file.

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
| 9 | [`ffn.py`](../app/vjepa_3/models/utils/ffn.py), [`norms.py`](../app/vjepa_3/models/utils/norms.py) — docstrings first, then `build_ffn` in `modules.py` | The non-positional flags (§9). Each file is under 120 lines |
| 10 | [`diagnostics.py`](../app/vjepa_3/diagnostics.py), then [`scripts/make_ladder_configs.py`](../scripts/make_ladder_configs.py) | How arms are measured and how they are defined (§9) |

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
| `models/utils/ffn.py` | **NEW** — `SquaredReLU`, the GLU gate table, `resolve_ffn_type` (§9.1) |
| `models/utils/norms.py` | **NEW** — `DynamicTanh`, `DynamicErf`, `build_norm_layer` (§9.2) |
| `diagnostics.py` | **NEW** — SIGReg / Epps-Pulley isotropy reading (§9.3) |
| `train.py` | 10 new `cfgs_model.get`, 1 new `cfgs_meta.get`; `EMBED_DIM_BY_MODEL_NAME` (§6); the diagnostic hook; single-rank guard (§6) |
| `utils.py` | 10 new `init_video_model` params; `init_opt` weight-decay exclusion (§4.3) |
| `wrappers.py`, `transforms.py`, `patch_embed.py`, `pos_embs.py`, `masks_dist.py` | unchanged copies |
| `models/utils/embeddings.py` | **DEAD** — nothing imports it. Removed from 2.1 after the fork snapshot; retained here only as an artifact. Safe to delete. |

Outside `app/vjepa_3/`:

| Path | What |
|---|---|
| `evals/feature_clustering_frozen/` | **NEW** — frozen-feature clustering probe: `cluster.py` (k-means, e-means adapter), `metrics.py`, `eval.py` (§9.4) |
| `configs/eval_3/vitb16/cluster-probe.yaml` | config for that probe |
| `configs/train_3/proxy/base-1gpu.yaml` | single-GPU, video-only base recipe (§9.5) |
| `configs/train_3/ladder/*.yaml` | **GENERATED** ablation arms — do not edit |
| `scripts/make_ladder_configs.py` | the generator, and the `ARMS` table that defines the ladder |

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
  ffn_type: null                     # gelu | relu2 | swiglu | geglu | reglu  (§9.1)
  pred_ffn_type: null                # same, for the predictor
  norm_type: null                    # layernorm | dyt | derf                 (§9.2)
  norm_out_type: null                # output norms only; follows norm_type
meta:
  sigreg_diag_freq: 0                # isotropy reading every N iters; 0 = off (§9.3)
optimization:
  lambda_warmup_frac: null           # [start, end] fractions of the run; null = 2.1's
                                     # fixed 15,000 -> 30,000 iterations (§6, §9.5)
```

`null` on the four new `model:` keys means "V-JEPA 2.1 behaviour" and takes the
unmodified code path, not an equivalent re-implementation of it.

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
| With `img_data` set and one rank, `int(world_size * img_rank_ratio)` is 0 and `img_total_batch_size % img_world_size` raises a bare `ZeroDivisionError` — so the stock configs cannot start on a single GPU. | **FIXED (message only)** — a `ValueError` explains that images and video train on disjoint ranks and points at the video-only proxy config. Joint training on one GPU is still not supported. |
| `n_registers` creates **no register tokens** anywhere (no parameter, no concatenation). It only tells RoPE to leave the last `n` tokens unrotated — and on the encoder the rotary table is still built for the full grid, so `n_registers > 0` raises a shape error in the forward pass. | **DOCUMENTED, not fixed** — pinned by `TestRegistersAreNotImplemented` on both trees. Do not use it as an ablation arm; real registers would be new work (§7). |
| `_rescale_blocks` divides `mlp.fc2.weight` by `sqrt(2·layer)`. For the MLP that is the output projection; for `SwiGLUFFN` `fc2` is the *gate-input* projection and the output (`fc3`) is left unscaled. | Carried over unchanged so `ffn_type: swiglu` stays bitwise equal to 2.1's `use_silu`. All GLU arms share it, so they remain comparable to each other; it is a confound when comparing GLU against the MLP arms. |
| The context ("dense") loss weight warms up between **fixed** iterations 15,000 and 30,000 (`Lambda_LinearWarmupHold` defaults; no config key). That is 5%–10% of the 300,000-iteration paper recipe, but on a short run it lands late or never: a 30,000-iteration run reaches full weight on its last step, a 9,000-iteration run trains with the context loss off throughout. | **FIXED (opt-in)** — `optimization.lambda_warmup_frac: [0.05, 0.1]` places the warm-up at the same fraction of any run length. Unset, the 2.1 iterations are used unchanged. The proxy base sets it; `TestContextLossWarmup` fails if it is removed. `train.py` logs the resolved iterations at start-up. |
| `meta: eval_freq` is in every config and read by nothing. | Carried over; listed in `KNOWN_DEAD_META_KEYS` so the meta-key test still catches new orphans. |

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
- **A real run of `train.py`.** `main()` has never executed end to end on
  this machine (`decord`, `webdataset`, `submitit`, `braceexpand` are not
  installed, and there is no GPU). Everything in this guide is verified at the
  model/unit level. The diagnostic hook in the training loop, the single-rank
  guard, and the proxy config's batch size and schedule are therefore
  **unexecuted**; expect to fix small things on the first launch.
- **Real register tokens** — see the `n_registers` row in §6.
- **SIGReg as a loss term** — only the reading is built (§9.3). H-JEPA found a
  marginal regulariser alone lets video models keep the background and drop
  the moving agent, so a loss version would have to sit alongside the EMA
  target, as a separate arm.
- **ViT-5 stabilisers** (LayerScale, QK-norm) — not built; recorded as the next
  candidates if the FFN/norm arms prove uninteresting.
- **e-means beyond the probe** — data curation or cluster-prediction targets.
  The adapter in `evals/feature_clustering_frozen/cluster.py` has not been
  executed (Faiss ≥ 1.16 not installed here).
- **`modded.yaml`** — the combined preset. `PRESET` in
  `scripts/make_ladder_configs.py` is empty until the ladder has results.

---

## 8. Verify

```bash
python3 -m pytest tests/models/ tests/distributed/ tests/probes/ -v
python3 scripts/make_ladder_configs.py --check
```

141 pass, 4 skip (3 CUDA-gated, 1 needs Faiss ≥ 1.16). CPU only, ~25s.

`vit_tiny` is `embed_dim=192 / num_heads=3 ⇒ head_dim=64` — the **same**
`head_dim` as ViT-L (1024/16) and ViT-g (1408/22), so `d_dim=h_dim=w_dim=20`
with 4 unrotated dims. The tiny model reproduces production rotary geometry
exactly. Do not widen it.

| Test file | Pins |
|---|---|
| `test_vjepa3_parity.py` | v3 ≡ 2.1 bitwise (video / image / masked), identical state_dict, the `EMBED_DIM` table, the `use_rope=False` breakage |
| `test_vjepa3_rope.py` | `build_omega` ≡ legacy bitwise across dtypes, the dead-channel counts, the fp32/bf16 negative result, isotropy validation, the hop chain |
| `test_vjepa3_string.py` | `P = I` at zero-init, orthogonality and `det = 1`, translation invariance, cls/register scores unchanged, antisymmetric gradient, activation-checkpointing agreement, predictor wiring, weight-decay exclusion |
| `test_vjepa3_configs.py` | every `model:` and `meta:` key has a reader in train.py *and* is forwarded to `init_video_model`; the proxy base is single-rank and video-only; ladder files equal the generator's output and each arm changes only its declared keys; `n_registers` is not a working feature |
| `test_vjepa3_ffn.py` | explicit `gelu`/`swiglu` ≡ the legacy switch ≡ 2.1, bitwise; `relu2` has GELU's parameters; each GLU gate is applied; conflicts and unknown names raise; encoder and predictor keys are independent |
| `test_vjepa3_norms.py` | DyT/Derf closed forms and published inits; inits survive `_init_weights`; every norm is replaced and `norm_out_type` is independent; `layernorm` ≡ default bitwise; fp32 under bf16 autocast; `alpha`/`shift` are never weight-decayed |
| `test_vjepa3_compose.py` | every pair of flag families trains; with all on, every parameter gets a gradient; activation checkpointing agrees with the plain path |
| `test_vjepa3_diagnostics.py` | equals LeJEPA Algorithm 1; Gaussian sits at the floor for any N; collapsed / mis-scaled / shifted / low-rank / heavy-tailed inputs score ≥ 20× the floor; immune to autocast |
| `tests/probes/test_cluster_probe.py` | purity and NMI against hand-computed values; k-means recovers separated blobs; the probe passes exactly the encoder kwargs the training path does; checkpoints load `strict=True` and a mismatched architecture is refused |
| `test_vjepa3_ddp_static_graph.py` | STRING — alone, and combined with SwiGLU + Derf + per-axis RoPE base in training mode — trains across a T=1 rank and a T=8 rank under `static_graph=True`, **with a negative control that must fail** |

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

---

## 9. The modded-ViT flags and the ablation ladder

Everything in §1–§5 is about *where* a token is. This section is about the
other layers of the block, how arms are measured, and how they are defined.

### What the evidence says before you spend GPU hours

| Flag | Source | What was shown | Prior for V-JEPA |
|---|---|---|---|
| `norm_type: derf` | [Stronger Normalization-Free Transformers](https://arxiv.org/pdf/2512.10938), CVPR 2026 | ViT-B ImageNet top-1 82.8 vs 82.5 (DyT) vs 82.3 (LayerNorm), supervised | Medium. Never tested with an EMA target or a regression loss onto normalised targets |
| `norm_type: dyt` | Transformers without Normalization, CVPR 2025 | Matches LayerNorm on supervised, MAE and DINO ViTs | Medium |
| `ffn_type: swiglu` | Standard in LLMs. [ViT-5](https://www.arxiv.org/pdf/2602.08071) deliberately leaves it out of its vision recipe | Small or no gain in ViTs across the papers found | Low–medium |
| `ffn_type: relu2` | No vision paper found that supports it. [Bug or Feature²](https://arxiv.org/pdf/2605.17659) reports squared activations do not beat GELU in vision | — | Low. Here because it was asked for and costs one line |
| e-means | Meta FAIR, Nov 2026, Faiss 1.16 `contrib/e_means` | Authors' claim: Lloyd's 25-epoch loss in 1.2–3 epochs | A faster k-means, not a better one. Expect the same probe scores |

None of these is a known win for this objective. That is the reason for a
ladder rather than a single "modded" config.

### 9.1 `ffn_type` — the feed-forward layer

```
gelu    fc2(GELU(fc1(x)))               V-JEPA 2.1 default
relu2   fc2(ReLU(fc1(x))²)              same shapes, same parameter count
swiglu  fc3(SiLU(fc1(x)) · fc2(x))      what `use_silu: true` already selected
geglu   fc3(GELU(fc1(x)) · fc2(x))
reglu   fc3(ReLU(fc1(x)) · fc2(x))
```

SwiGLU was already in V-JEPA 2.1 behind `use_silu`; `ffn_type` gives the choice
a name and adds the others. `ffn_type: null` defers to `use_silu` through the
original branch in `Block.__init__`. Setting `use_silu: true` together with a
non-`swiglu` `ffn_type` raises, because two keys asking for different layers
would mislabel an arm.

### 9.2 `norm_type` / `norm_out_type` — DyT and Derf

LayerNorm measures each token (mean and variance over channels) and rescales
by what it measured. DyT and Derf measure nothing: each is a fixed S-shaped
squashing function with one learnable input scale per layer, then LayerNorm's
usual per-channel `weight` and `bias`.

```
dyt    weight · tanh(alpha · x)         + bias     alpha = 0.5
derf   weight · erf (alpha · x + shift) + bias     alpha = 0.5, shift = 0
```

- `norm_type` replaces the two pre-norms in every block, in **both** encoder
  and predictor.
- `norm_out_type` replaces the output norms: the encoder's four
  deep-supervision norms and the predictor's final norm. It follows `norm_type`
  unless set. It is a separate key because those outputs feed the loss: the
  predictor regresses onto LayerNorm-standardised targets, and neither paper
  tested a bounded output layer in that position. If `n1-dyt` or `n2-derf`
  trains badly, `norm_out_type: layernorm` is the first thing to try.
- The `F.layer_norm` calls in `train.py` standardise the *targets*. They are
  part of the loss and are never replaced.

Three properties fall out of the parameter shapes rather than from extra code,
and each is pinned by a test: `alpha`/`shift` are 1-D, so `init_opt` never
weight-decays them; `_init_weights` does not touch them; and under bf16
autocast the squashing runs in fp32 (an fp32 parameter times a bf16 activation
promotes to fp32), the same precision LayerNorm gets.

### 9.3 The isotropy reading (`meta.sigreg_diag_freq`)

LeJEPA argues the ideal embedding distribution is an isotropic Gaussian and
measures the distance to it with SIGReg. Here that statistic is **logged, not
optimised**: every `sigreg_diag_freq` iterations, rank 0 takes the last
deep-supervision level of the target encoder (before the loss-side
`F.layer_norm`), samples 4096 tokens, and appends a row to
`<folder>/diag_r0.csv`.

How to read the two columns:

| | Perfect isotropic Gaussian | Otherwise |
|---|---|---|
| `sigreg_raw` — features as they are | ≈ 1.06 at any sample size | grows linearly with N |
| `sigreg_std` — each dimension z-scored first | ≈ 0.58 | grows linearly with N |

`sigreg_raw` high with `sigreg_std` near its floor means the shape is fine and
only the scale or mean is off. Both high means dimensions are correlated or
the rank is low — the direction collapse takes. At N = 4096 a fully collapsed
representation scores in the thousands. Reading-to-reading noise is roughly
±7%.

Two limits. Tokens from one clip are not independent, so this is a relative
reading across arms and across training time, not a calibrated test. And
random projections of high-dimensional data look Gaussian easily, so it will
not see non-Gaussianity confined to single coordinates.

### 9.4 The clustering probe (`evals/feature_clustering_frozen`)

Embeds each clip of a labelled set as one vector (mean over tokens), clusters
without labels, and scores the clusters against the classes.

```bash
python -m evals.main --fname configs/eval_3/vitb16/cluster-probe.yaml --devices cuda:0
```

- It reads the architecture from the run's own `params-pretrain.yaml` and loads
  the checkpoint with `strict=True`, so it cannot evaluate a different model
  than the one that was trained. It builds from `app.vjepa_3.models`, so —
  unlike the other evals (§7) — it uses the correct RoPE.
- Quote **NMI**, not purity, when comparing settings: purity rises with the
  number of clusters by construction.
- `clusterers: [kmeans, emeans]` runs both and reports `quantization_error`
  (the objective both minimise) and wall time for each. That is the
  like-for-like comparison of e-means against k-means on your features.
- It is coarse. Mean-pooling discards where and when things happen, so it
  mostly rewards scene and appearance. It cannot stand in for the motion
  benchmarks, and a positional-encoding arm may barely move it.

### 9.5 The ladder

`configs/train_3/proxy/base-1gpu.yaml` is a single-GPU, video-only version of
the ViT-B recipe. `scripts/make_ladder_configs.py` generates one config per
arm from it; its `ARMS` table is the definition of the ladder.

| Arm | Changes vs `a0-control` | Warm start from 2.1 |
|---|---|---|
| `a0-control` | nothing | yes |
| `p1-theta100` | `rope_theta: 100.0` | yes |
| `p2-theta-temporal` | `rope_theta: [100, 10000, 10000]` | yes |
| `p3-string` | `use_string: true` | yes |
| `f1-swiglu` | `ffn_type`, `pred_ffn_type: swiglu` | **no** |
| `f2-relu2` | `ffn_type`, `pred_ffn_type: relu2` | **no** |
| `n1-dyt` | `norm_type: dyt` | **no** |
| `n2-derf` | `norm_type: derf` | **no** |

```bash
python3 scripts/make_ladder_configs.py            # regenerate after editing the base or ARMS
python -m app.main --fname configs/train_3/ladder/n2-derf.yaml --devices cuda:0
```

Rules for reading the results:

1. **Two families, two controls.** A positional arm can start from a 2.1
   checkpoint; an FFN or norm arm cannot, because the same weights would
   compute a different function. Compare warm-started arms against a
   warm-started control and from-scratch arms against a from-scratch control.
   Never across.
2. **The base is a screening recipe; freeze it across arms.** Values marked
   `SCREEN` in the base were set from measurements on one Quadro RTX 8000
   (its header lists each one and why): `float16`, activation checkpointing
   off, 30 epochs (9,000 iterations, ~3.5 h per arm), `lambda_warmup_frac`
   so the context loss is on for 90% of the run, and an untuned `lr` of
   2e-4. A 9,000-iteration run ranks arms; it can under-rate a change that
   only pays off late, so checkpoints are kept at epochs 10/20/30 to see
   whether a ranking is stable, and a winner is confirmed with a longer run
   before it goes into the preset.
3. **One seed is one sample.** On short proxy runs, differences of the size
   reported in the papers above (a few tenths of a point) are well inside
   seed noise. Re-run the control with a second seed before believing any
   arm that wins by a small margin.
4. **Then the preset.** Put the winning arm names in `PRESET` and regenerate;
   `ladder/modded.yaml` is written as the union of their overrides, and the
   script refuses arms that set the same key differently.

There is no registers arm: see `n_registers` in §6.

