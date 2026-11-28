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
**Cayley-STRING** puts a learned orthogonal basis in front of this split (§4).
It does not let the axes share rotation planes: the rotation keeps its
per-axis blocks and frequencies, and the basis can be folded exactly into each
layer's q/k weights, so the arm tests optimisation, not a larger function
class (audit S-01, S-03).

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
θ=1000 → 6/10 live, θ=100 → 8/10. Those counts are for the encoder's 10
frequency pairs under this document's own ">0.2 rad" criterion; the predictor
has 5 pairs per axis, and by "wavelength shorter than the axis" the count
barely moves (audit R-08, R-10). The supporting evidence is one heuristic from
2-D image RoPE (RoPE-Mixed); the STRING paper keeps 10,000 for plain RoPE, and
nothing cited tests a smaller base on a temporal axis (audit R-11). Note even
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
| `regularizers.py` | **NEW** — SIGReg and VISReg as optional loss terms on the encoder's tokens (§9.7) |
| `pretrained.py` | **NEW** — loading a released V-JEPA 2.1 encoder; what can and cannot be carried over (§9.8) |
| `masks.py` | **NEW** — options on a `mask:` entry: hide whole time-steps, cut token lists at random instead of from the end of the clip (§9.10), and send a weight per hidden token (§9.11) |
| `change.py` | **NEW** — how much of a hidden patch could simply be copied from a shown frame, and the weighted loss built from it (§9.11). Also used by `scripts/measure_change.py` |
| `train.py` | 10 new `cfgs_model.get`, 3 new `cfgs_meta.get`; `EMBED_DIM_BY_MODEL_NAME` (§6); the diagnostic hook; single-rank guard (§6); the regulariser term (§9.7); encoder warm start, freeze and lr scale (§9.8); gradient clipping; atomic checkpoints and `persistent_workers` (§6) |
| `utils.py` | 10 new `init_video_model` params; `init_opt` weight-decay exclusion (§4.3) |
| `wrappers.py`, `transforms.py`, `patch_embed.py`, `pos_embs.py`, `masks_dist.py` | unchanged copies |
| `models/utils/embeddings.py` | **DEAD** — nothing imports it. Removed from 2.1 after the fork snapshot; retained here only as an artifact. Safe to delete. |

Outside `app/vjepa_3/`:

| Path | What |
|---|---|
| `evals/feature_clustering_frozen/` | **NEW** — frozen-feature clustering probe: `cluster.py` (k-means, e-means adapter), `metrics.py`, `eval.py` (§9.4) |
| `configs/eval_3/vitb16/cluster-probe.yaml` | config for that probe |
| `evals/video_tracking_frozen/` | **NEW** — frozen-feature object-tracking probe on DAVIS 2017: `propagate.py` (label propagation), `metrics.py` (J and F), `davis.py` (dataset reader), `eval.py` (§9.6) |
| `configs/eval_3/vitb16/davis-tracking.yaml` | config for that probe |
| `configs/train_3/proxy/base-1gpu.yaml` | single-GPU, video-only base recipe (§9.5) |
| `configs/train_3/ladder/*.yaml` | **GENERATED** ablation arms — do not edit |
| `scripts/make_ladder_configs.py` | the generator, and the `ARMS` table that defines the ladder; `--local` writes copies with one machine's paths (§9.5) |
| `scripts/preflight_arm.py` | **NEW** — one real training step of an arm against the control; run before any arm goes to the GPU (§9.9) |
| `scripts/compare_tracking.py` | **NEW** — tracking scores against a control with error bars and a training-noise yardstick (§9.9) |
| `tests/train/` | **NEW** — the real `train.main()` end to end on CPU (`harness.py`), and the launcher's exit status |
| `V3_STRING/PROGRESS.md` | **NEW** — the running record: what was run, what it showed, what is next |
| `V3_STRING/LADDER_AUDIT.md` | **NEW** — every finding of the theory-vs-implementation audit of the first ladder |

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
loss:
  reg_type: null                     # sigreg | visreg, added to the loss       (§9.7)
  reg_weight: 0.0                    # must be > 0 when reg_type is set
  reg_num_slices: 1024               # random 1-D projections per step
  reg_max_tokens: 4096               # tokens sampled per step
mask:                                # per entry of the list              (§9.10)
- hide_time_steps: 0                 # k > 0: hide k whole time-steps instead of blocks
  target_weight: none                # unexplained: weight each hidden token (needs hide_time_steps > 0)
  target_weight_full: 0.064          # the change that earns weight 1
  truncation: tail                   # random: use the whole clip evenly
  max_temporal_keep: 1.0             # 2.1 key; < 1: context from the start of the clip only
data:
  fps: 4                             # 2.1 key; frames sampled per second
  persistent_workers: false          # read since Nov 2026 (§6); 2.1 ignored it
meta:
  sigreg_diag_freq: 0                # isotropy reading every N iters; 0 = off (§9.3)
  pretrained_encoder: null           # path of a released 2.1 file              (§9.8)
  pretrained_encoder_key: ema_encoder  # target_encoder for ViT-g/G and our own files
optimization:
  lambda_warmup_frac: null           # [start, end] fractions of the run; null = 2.1's
                                     # fixed 15,000 -> 30,000 iterations (§6, §9.5)
  clip_grad: null                    # max global gradient norm; null = none, as in 2.1
  freeze_encoder_iters: 0            # encoder takes no step for N iterations   (§9.8)
  encoder_lr_scale: 1.0              # encoder lr as a multiple of the schedule (§9.8)
```

Every key above defaults to V-JEPA 2.1's behaviour. With all of them at the
default, the v3 training loop produces the same checkpoint as the 2.1 loop,
bit for bit (`tests/train/test_vjepa3_train_loop.py`).

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
| `_rescale_blocks` divides `mlp.fc2.weight` by `sqrt(2·layer)`. For the MLP that is the output projection; for `SwiGLUFFN` `fc2` is the *value* branch (the gate input is `fc1`) and the output (`fc3`) is left unscaled. | Carried over unchanged so `ffn_type: swiglu` stays bitwise equal to 2.1's `use_silu`. All GLU arms share it, so they remain comparable to each other; it is a confound when comparing GLU against the MLP arms. |
| The context ("dense") loss weight warms up between **fixed** iterations 15,000 and 30,000 (`Lambda_LinearWarmupHold` defaults; no config key). That is 5%–10% of upstream's 300,000-iteration config (the paper reports 135,000 + 12,000), but on a short run it lands late or never: a 30,000-iteration run reaches full weight on its last step, a 9,000-iteration run trains with the context loss off throughout. | **FIXED (opt-in)** — `optimization.lambda_warmup_frac: [0.05, 0.1]` places the warm-up at the same fraction of any run length. Unset, the 2.1 iterations are used unchanged. The proxy base sets it; `TestContextLossWarmup` fails if it is removed. `train.py` logs the resolved iterations at start-up. |
| `meta: eval_freq` is in every config and read by nothing. | Carried over; listed in `KNOWN_DEAD_META_KEYS` so the meta-key test still catches new orphans. |
| `data: persistent_workers` is in every config and read by nothing, so the loader respawns all workers each time the data runs out (about 110 s per refill on the proxy recipe, 19 times a run). | **FIXED (opt-in)** — `train.py` reads the key and passes it on; absent or `false` is the old behaviour. The proxy base sets `false` on purpose: the first ladder ran that way, and kept-alive workers draw a different augmentation and mask stream. Not yet exercised with `true` on the server. |
| Checkpoints are written in place and a failed write is only logged, so a crash or a full disk mid-write leaves a truncated `latest.pth.tar` and the run can end "successfully" with nothing to resume from. | **FIXED** — written to `<name>.tmp` and renamed (atomic); a failed write of the final checkpoint raises. |
| `app/main.py` starts its worker processes and returns 0 whatever happens to them, so a shell loop over configs carries on after a crashed run. | **FIXED** — the launcher waits for every rank and exits 1 if any failed. |
| The cooldown configs ship four Gram-loss keys (`computing_gram_loss`, `gram_HighRes`, `gram_ckpt`, `gram_loss_weight`) and `loss: reg_coeff`; the released `train.py` has no Gram loss and reads none of them. | Carried over; listed in `KNOWN_DEAD_LOSS_KEYS` so the loss-key test still catches new orphans. |

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
- **CUDA / float16 coverage in tests.** `train.main()` now runs end to end in
  the test suite, but on CPU in float32. The float16 + GradScaler path is
  exercised only by the real runs on the server (seven finished; see
  `PROGRESS.md`). The regulariser, warm-start, clipping and
  `persistent_workers: true` paths have **not** run on the GPU yet.
- **Learnable RoPE frequencies** (RoPE-Mixed) — the audit's suggested
  replacement for the STRING arm, whose learned basis adds no capacity while
  the frequencies stay frozen (audit S-01, S-02). Not built.
- **A norm-free arm that can train** — `n1-dyt` collapsed. A fair retest needs
  an initialisation that does not rely on LayerNorm for its starting scale
  (audit N-09), float32 or clipping, and a pre-flight pass (§9.9). Not built;
  `clip_grad` and `norm_out_type: layernorm` exist as ingredients.
- **A projector for the regulariser** — both papers regularise the output of
  a projector MLP, not the backbone features (§9.7).
- **Exact resume** — a resumed run differs from an uninterrupted one at the
  level of a different seed, because RNG state is not checkpointed (audit L-08).
- **Real register tokens** — see the `n_registers` row in §6.
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
python3 -m pytest tests/ -q
python3 scripts/make_ladder_configs.py --check
```

259 pass, 4 skip (3 CUDA-gated, 1 needs Faiss ≥ 1.16). CPU only, about 100 s
(40 s of that is one test: the v3 loop against the 2.1 loop on ViT-L).

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
| `tests/probes/test_tracking_probe.py` | J and F against closed-form cases; label propagation tracks a synthetic object and loses it when the features carry no signal; frame/clip feature extraction keeps token order; `main()` end to end on a synthetic DAVIS-layout folder with a saved checkpoint, and a mismatched checkpoint is refused |
| `tests/probes/test_cluster_probe.py` | purity and NMI against hand-computed values; k-means recovers separated blobs; the probe passes exactly the encoder kwargs the training path does; checkpoints load `strict=True` and a mismatched architecture is refused |
| `test_vjepa3_regularizers.py` | SIGReg equals the logged reading divided by N; VISReg equals the paper's listing; each term responds to its own failure (scale, centre, heavy tails); the stop-gradient on the scale is present; gradients are finite under exact collapse; half-configured settings raise |
| `test_vjepa3_pretrained.py` | the architecture built for a released ViT-B equals the one Meta's loader builds (158 tensors); the warm-start arms build it too and the FFN/norm arms do not; loading is strict and `weights_only` |
| `tests/train/test_vjepa3_train_loop.py` | the real `train.main()`: defaults ≡ the 2.1 loop bitwise; each regulariser trains, is logged, and touches only the encoder; warm start, freeze, lr scale and resume; gradient clipping; atomic checkpoints and a fatal failed final save; `persistent_workers` reaches the loader |
| `test_vjepa3_change.py` | weights on a scene with a known answer: the object counts and the background does not, one shown frame that explains a patch is enough, what was never in view counts fully; the weighted loss equals the plain one at weight 1, sums in float32, and pushes in proportion to the weight; the collator sends the weights as a fourth item |
| `test_vjepa3_masks.py` | default configs get 2.1's own collator, and the v3 generator at defaults reproduces it exactly; a random cut keeps the budget, stays sorted, never shows a hidden position, and evens out the time-steps; whole-time-step masks hide exactly k steps, differently per clip |
| `tests/train/test_launcher.py` | `app/main.py` exits non-zero when a rank crashes |
| `tests/scripts/test_ladder_tools.py` | the comparison tool resamples by video and needs both noise sources cleared before calling an arm "clear"; the pre-flight check flags `n1-dyt` and passes healthy arms |
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
  predictor regresses onto LayerNorm-standardised targets. (DyT's MAE
  experiment does replace the output norm before a regression head; Derf
  reports no such case.) `n1-dyt` collapsed in the first ladder (`PROGRESS.md`);
  `norm_out_type: layernorm` alone is not expected to fix that, because the
  cause is the starting scale inside the blocks (audit N-09).
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

| Arm | Changes vs the base | Encoder loadable from a 2.1 file | Compare with |
|---|---|---|---|
| `a0-control` | nothing | yes | – |
| `p1-theta100` | `rope_theta: 100.0` | yes | `a0-control` |
| `p2-theta-temporal` | `rope_theta: [100, 10000, 10000]` | yes | `a0-control` |
| `p3-string` | `use_string: true` | yes | `a0-control` |
| `f1-swiglu` | `ffn_type`, `pred_ffn_type: swiglu` | **no** | `a0-control` |
| `f2-relu2` | `ffn_type`, `pred_ffn_type: relu2` | **no** | `a0-control` |
| `n1-dyt` | `norm_type: dyt` | **no** | `a0-control` |
| `n2-derf` | `norm_type: derf` | **no** | `a0-control` |
| `s1-sigreg` | `loss.reg_type: sigreg`, `reg_weight: 0.1` | yes | `a0-control` |
| `s2-visreg` | `loss.reg_type: visreg`, `reg_weight: 0.05` | yes | `a0-control` |
| `n3-dyt-sigreg` | `norm_type: dyt` + SIGReg at 0.1 | no | `a0-control`, `n1-dyt`, `s1-sigreg` |
| `n4-derf-sigreg` | `norm_type: derf` + SIGReg at 0.1 | no | `a0-control`, `n3-dyt-sigreg` |
| `t1-future` | `mask.0.max_temporal_keep: 0.5` (§9.10) | yes | `a0-control` |
| `t2-fps12` | `data.fps: 12` (§9.10) | yes | `a0-control` |
| `t3-frames` | `mask.0.hide_time_steps: 6` (§9.10) | yes | `a0-control` |
| `t5-frames-weighted` | `t3-frames` + `mask.0.target_weight: unexplained` (§9.11) | yes | `t3-frames` |
| `w7-warm-frames-weighted` | `w5-warm-frames` + `mask.0.target_weight: unexplained` | yes | `w5-warm-frames` |
| `t4-whole-clip` | `truncation: random` on both masks (§9.10) | yes | `a0-control` |
| `w0-warm-control` | released ViT-B encoder, 900 frozen iterations, encoder lr ×0.1 | yes | – |
| `w2-warm-theta-temporal` | `w0` + `rope_theta: [100, 10000, 10000]` | yes | `w0-warm-control` |
| `w3-warm-future` | `w0` + `mask.0.max_temporal_keep: 0.5` | yes | `w0-warm-control` |
| `w4-warm-fps12` | `w0` + `data.fps: 12` | yes | `w0-warm-control` |
| `w5-warm-frames` | `w0` + `mask.0.hide_time_steps: 6` | yes | `w0-warm-control` |
| `w6-warm-whole-clip` | `w0` + `truncation: random` on both masks | yes | `w0-warm-control` |

"Loadable" says the arm keeps the encoder's parameters. Only the `w` arms
actually start from a released file; every other arm trains from scratch.
Results of the first eight are in [PROGRESS.md](PROGRESS.md).

```bash
python3 scripts/make_ladder_configs.py            # regenerate after editing the base or ARMS
python3 scripts/preflight_arm.py configs/train_3/ladder/s1-sigreg.yaml   # before the GPU (§9.9)

# on the training machine: copies with real paths, in a folder git ignores
python3 scripts/make_ladder_configs.py --local configs/train_3/local \
    --folder-root /runs/pretrain_3/ladder --data /data/train_paths.csv
python -m app.main --fname configs/train_3/local/s1-sigreg.yaml --devices cuda:0
```

`--seed N` with `--local` writes every arm again as `<arm>-seedN` with
`meta.seed: N`, for repeat runs. The tracked files keep placeholder paths;
never edit them by hand (the first ladder did, and the edited copies could not
be audited: audit L-25).

Rules for reading the results:

1. **Two families, two controls.** A positional arm can start from a 2.1
   checkpoint; an FFN or norm arm cannot, because the same weights would
   compute a different function. Compare warm-started arms against a
   warm-started control and from-scratch arms against a from-scratch control.
   Never across.
2. **The base is a screening recipe; freeze it across arms.** Values marked
   `SCREEN` in the base were set from measurements on one Quadro RTX 8000
   (its header lists each one and why): `float16`, activation checkpointing
   off, 30 epochs (9,000 iterations, about 5 h per arm as measured), `lambda_warmup_frac`
   so the context loss is on for 90% of the run, and an untuned `lr` of
   2e-4. A 9,000-iteration run ranks arms; it can under-rate a change that
   only pays off late, so checkpoints are kept at epochs 10/20/30 to see
   whether a ranking is stable, and a winner is confirmed with a longer run
   before it goes into the preset.
3. **One seed is one sample.** On short proxy runs, differences of the size
   reported in the papers above (a few tenths of a point) are well inside
   seed noise. Re-run the control with a second seed before believing any
   arm that wins by a small margin, and read scores through
   `scripts/compare_tracking.py` (§9.9), which reports both the test-set
   range and the gap between controls.
4. **Then the preset.** Put the winning arm names in `PRESET` and regenerate;
   `ladder/modded.yaml` is written as the union of their overrides, and the
   script refuses arms that set the same key differently.

There is no registers arm: see `n_registers` in §6.

### 9.6 The tracking probe (`evals/video_tracking_frozen`)

The reading the ladder is ranked by. The objects of a DAVIS 2017 validation
video are outlined in its first frame; the probe tracks them through the
video using only the similarity of frozen patch features between frames
(label propagation, Jabri et al. 2020). Nothing is trained. The score is
**J&F-Mean**: mask overlap (J) and outline accuracy (F), averaged over every
object of the 30 videos.

```bash
python -m evals.video_tracking_frozen.eval --run <run folder> --davis <DAVIS root>
python -m evals.video_tracking_frozen.eval --run <run folder> --davis <DAVIS root> --mode clip
python -m evals.video_tracking_frozen.eval --run <run folder> --davis <DAVIS root> --random-init
```

- **Protocol** is V-JEPA 2.1's (paper §7, §C.3): short side 480 px, 15
  context frames, circle neighbourhood of radius 12, top-5, temperature 0.2.
  The paper reports 67.0 for its distilled ViT-B. A 9,000-iteration proxy run
  will be far below that; use the probe to rank arms trained identically.
- **`--random-init`** scores an untrained encoder of the same architecture.
  Label propagation gets some overlap from the spatial neighbourhood alone,
  so this is the floor a trained arm has to beat. Run it once per arm family.
- **`--mode frame`** (default) feeds each frame alone, repeated to fill one
  tubelet. **`--mode clip`** feeds 16-frame windows as in training, so the
  temporal position encoding is exercised, at the cost of one feature map per
  two frames. Report both for the positional arms (`p1`–`p3`): in `frame`
  mode every token sits at temporal position 0.
- The report also splits objects into small / medium / large by the share of
  the first frame they cover (below 1%, 1–5%, above 5%). That split is ours,
  not DAVIS's.
- It reads the architecture from the run's `params-pretrain.yaml` and loads
  `target_encoder` with `strict=True`, like the clustering probe.
- **`--released <file> --out <folder>`** scores one of Meta's released
  V-JEPA 2.1 encoders instead of a run of ours (the file keeps its released
  name, e.g. `vjepa2_1_vitb_dist_vitG_384.pt`). This is the yardstick check:
  the paper reports 67.0 for that model, and our protocol differs slightly
  (audit L-18), so the number tells us how far this probe sits from the
  paper's. Not yet run.
- **Status:** run on the real DAVIS 2017 val set for seven trained arms
  (`PROGRESS.md`). The audit checked J and F against a reference copy of the
  official metrics on 180 mask pairs and the propagation against DINO's
  (L-16, L-17). Still open: the installed official toolkit on real
  predictions, and `--mode clip` at full size on the GPU.

### 9.7 Regularisers on the loss (`loss.reg_type`)

V-JEPA has no loss term that forbids collapse. LeJEPA's SIGReg and VISReg add
one: both pull the embedding distribution toward an isotropic Gaussian through
many random 1-D projections. `app/vjepa_3/regularizers.py` implements both as
an **addition** to the existing loss:

```
loss = L_pred + λ·L_ctx + reg_weight · L_reg(encoder tokens)
```

How this differs from the papers, and why:

| | Papers | Here |
|---|---|---|
| EMA teacher | removed | **kept**. Our H-JEPA notes record that a marginal regulariser alone lets a video model keep the background and drop the moving object |
| What is regularised | one pooled vector per image, after a projector MLP | **patch tokens** of the context encoder's last output level, no projector. A batch is 16 clips: too few vectors to shape a distribution |
| Samples per step | 128–512 images (VISReg loses 2 points going from 512 to 128) | up to 4,096 tokens, drawn across the batch. Tokens of one clip are not independent; neither paper tested that |
| Weight | about 10× the prediction loss | first guesses, set so each regulariser adds about a third of the main loss's gradient on the encoder at the first step (SIGReg 0.1, VISReg 0.05; measured with §9.9 on synthetic clips) |
| SIGReg value | multiplied by the sample count N | **not** multiplied, so a weight means the same at any token count. Reference values: 0 Gaussian, 0.41 all tokens at one point, 0.82 spread far too wide |

`<folder>/reg_r0.csv` logs the main loss and the regulariser separately at
every iteration; `loss` in `log_r0.csv` is their weighted sum and is not
comparable with an arm that has no regulariser.

What to expect, from what we have measured: the baseline's features are far
from the target in *offset and per-dimension scale* (`sigreg_raw` ≈ 0.4 per
sample) but close in *shape* (`sigreg_std` ≈ 0.007). So the regulariser will
mostly centre and rescale the dimensions. Across the six healthy arms of the
first ladder the isotropy reading showed no relation to the tracking score
(`PROGRESS.md`), so a gain in tracking is a hypothesis, not an expectation.
The papers' own dense-feature result is their stated weak spot (VISReg: 30.2
mIoU on ADE20K against MoCo v3's 31.7).

### 9.8 Starting from a released encoder (`meta.pretrained_encoder`)

The first ladder trains ViT-B from scratch on 7,400 videos and reaches a
tracking score of 35; the paper's distilled ViT-B scores 67. To test a change
on top of a model that already works:

```yaml
meta:
  pretrained_encoder: /ckpt/vjepa2_1_vitb_dist_vitG_384.pt
optimization:
  freeze_encoder_iters: 900     # predictor trains alone first
  encoder_lr_scale: 0.1         # then the encoder moves slowly
```

- **Only the encoder is loaded.** The released ViT-B was distilled from ViT-G
  and its predictor outputs the teacher's 1664-wide features at one level; the
  loop here predicts the encoder's own width at four levels. So the predictor
  starts untrained, which is what the two guards are for. The distillation
  loop itself is not in the released code.
- **The encoder fits exactly**: 158 of 158 tensors with the same shapes as
  the encoder the proxy recipe builds, for every arm that keeps the encoder's
  parameters (`test_vjepa3_pretrained.py`).
- **A different question from the first ladder.** These runs continue a
  distilled model with a self-supervised objective. `w0-warm-control` exists
  to answer the first question: does the score survive that at all? Run and
  score it before any other `w` arm. If it does not hold, the guards (both
  first guesses) need tuning before the family means anything.
- **Not verified:** no released file has been opened by this code. The key
  names come from Meta's own loader (`src/hub/backbones.py`).

### 9.9 Two tools: check before the GPU, compare after

```bash
python3 scripts/preflight_arm.py configs/train_3/ladder/<arm>.yaml
```

runs one real training step of the arm and of the control on identical
synthetic clips (CPU, about 10 s per arm at ViT-B) and compares the gradient
each produces. `n1-dyt`, which collapsed after five GPU-hours, fails this in
one step: its encoder gradient is about 50 times smaller than the control's
and a third of its elements are below Adam's epsilon. A PASS means "starts in
the same regime as the control", nothing more: one step, float32, synthetic
data.

```bash
python3 scripts/compare_tracking.py --runs <runs folder> \
    --control a0-control --control a0-control-s2 [--mode clip]
```

prints each arm's tracking score against the control with a 95% range from
resampling the 30 test videos, and the gap between the two controls as the
training-noise yardstick. An arm is marked "clear" only when its range
excludes zero **and** its difference exceeds that gap.

### 9.10 Time-focused arms: what is hidden, and how video is sampled

These two change no network code. They change the task and the input, which
is where the objective and the goal (following things through time) part ways.

**What the recipe does today**, measured on the real mask generator at the
recipe's size (16 frames = 8 time-steps of 256 tokens, batch 16; share of each
time-step's tokens, averaged over 3,200 clips):

| | step 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mask 0, visible to the encoder | 39% | 39% | 38% | 36% | 29% | 16% | 7% | 3% |
| mask 0, prediction target | 61% | 61% | 61% | 61% | 60% | 49% | 24% | 7% |
| mask 1, visible to the encoder | 19% | 16% | 10% | 6% | 3% | 2% | 1% | 1% |
| mask 1, prediction target | 81% | 81% | 81% | 81% | 81% | 80% | 61% | 20% |

Two things follow:

1. **The same spatial blocks are hidden in every frame** (`temporal_scale:
   [1.0, 1.0]`), so the task is mostly "fill in space"; nothing requires
   following an object from one frame to the next.
2. **The end of every clip is largely unused**: 26% (mask 0) and 22% (mask 1)
   of all tokens are neither shown nor asked about, almost all of them in the
   last two time-steps. The generator cuts every token list in a batch to the
   length of the shortest one, and the lists are sorted by time, so the cut
   always removes the latest tokens. This is upstream behaviour
   (`src/masks/multiseq_multiblock3d.py`, frozen here), present in V-JEPA 2.1
   as released; a larger batch makes the cut deeper. `t4-whole-clip` changes
   it.

**`t1-future`** sets `max_temporal_keep: 0.5` on mask 0 only: its context
comes from the first four time-steps and the predictor must produce the later
ones (targets per time-step become 61 / 61 / 61 / 61 / 100 / 100 / 99 / 48%).
Mask 1 is unchanged, so the encoder still sees every time position. The risk:
Meta hides the same region in all frames precisely so the model cannot copy
from a neighbouring frame; predicting the future of a static scene is easy,
and an easy task teaches little.

**`t2-fps12`** samples frames at 12 per second instead of 4, so 16 frames
span 1.3 s instead of 4 s. The tracking test runs on consecutive 24-fps
frames; at 4 fps the model never trains on displacements that small. The
cost: less long-range context per clip. Note the pre-flight check cannot see
this arm (its synthetic clips have no frame rate).

**`t3-frames`** hides whole moments instead of regions. For mask 0, six of
the eight time-steps, chosen at random per clip, are hidden completely and
predicted from the two that are shown (`hide_time_steps: 6`,
`app/vjepa_3/masks.py`). The idea: a video is a set of samples of a continuous
world, so ask the model for the moments it was not given. Six was chosen to
match the control's budget: two visible time-steps are 512 tokens and the
control's mask 0 shows about 504, so the arm changes how the visible quarter
is arranged, not how much is shown. Per time-step it shows 25% and asks about
75%, evenly, with nothing unused. Known risks: Meta and VideoMAE chose against
frame masking because neighbouring frames are similar enough to copy from; and
one time-step is two frames, so frames are hidden in pairs. This is the
settings-only half of the idea. The other half (time positions in seconds, and
several frame rates, so the model can be asked about in-between moments) is
not built.

**`t4-whole-clip`** fixes finding 2 above and nothing else: when a batch's
token lists are cut to a common length, a random subset is kept instead of the
first entries (`truncation: random` on both masks). The same number of tokens
is shown and asked about; every time-step now gets the same share (mask 0: 26%
shown and 48% asked in each step, against 39% → 3% and 61% → 7% from first
step to last in the control).

All four exist from scratch (`t1`–`t4`, against `a0-control`) and on the
released encoder (`w3`–`w6`, against `w0-warm-control`). Read them in
`--mode clip`: single-frame mode cannot see a change that is about time.

### 9.11 Weighted targets: do not ask for what can be copied

**The gap.** V-JEPA's authors tried masks that cover less of the clip in time,
found them worse on action classification, called the task trivial, and kept
masks that cover the whole clip (Bardes et al., 2024, section E.4). Three
things were left open:

1. *Trivial was treated as a property of the mask. It is a property of each
   patch.* A patch that looks the same in a frame the model was shown can be
   copied; one where something moved cannot. Covering the whole clip removes
   the first kind by never showing any place at another time, which removes
   the second kind too.
2. *Every hidden patch counts the same.* With `loss_exp: 1.0` the slope of
   the loss is +1 or −1 whatever the error, so a copyable patch pushes on the
   model as hard as a hard one.
3. *The camera was ignored.* "The same region in every frame" hides the same
   content only when the camera is still.

All of it was judged on classification, never on a per-patch task.

**What we measured** (`scripts/measure_change.py`, `PROGRESS.md` §8.1). In the
training videos, at a 0.25 s gap, 57% of patches are explained by the one
motion most of the frame shares; at a one-frame gap, 80%. `t3-frames`, which
hides whole time-steps and counts every patch equally, is the one time-focused
arm that came out clearly worse than the control (−3.8 in 16-frame mode).

**What `target_weight: unexplained` does.** On a mask with
`hide_time_steps > 0`, each hidden token gets a weight in [0, 1]:

    for each shown time-step:
        undo the shift most of the frame shares        change.shared_shift
        measure what still differs, patch by patch     change.patch_change
        scale it to [0, 1]                             change.unexplained_weight
    weight = the smallest of those

and the loss on that mask becomes `sum(w · |error|) / sum(w)`
(`change.weighted_error`). The other mask keeps V-JEPA's blocks and its plain
loss. The weights are computed in the data-loading workers by
`MaskCollatorV3` and travel with the batch; `weight_r0.csv` in the run folder
records their mean every iteration.

On a synthetic clip (camera pans, one square moves on its own) the square is
16 of 256 patches: 6% of the push with the plain loss, 68% with the weights.
On DAVIS the annotated objects are 30% of the patches and receive about half
of the weight.

**The test.** `t5-frames-weighted` against `t3-frames`: same seed, same mask,
only the weights differ. If the weighted run is no better, copyable patches
are not why hiding whole moments hurts.

**Limits.** One shift does not describe zoom, rotation or depth (little is
explained there and the weights stay near 1, which is `t3-frames`'
behaviour). "Unexplained" is not "an object". Only the first frame of each
two-frame token is compared. The scale 0.064 was read off DAVIS, which is
also the test set. The weights cost about 0.6 s of one data-loading worker per
clip; whether 24 workers keep up with the GPU is not measured.

**Earlier work.** MGMAE (Huang et al., 2023) and MGM (Fan et al., 2023) use
motion to decide where a mask is placed in pixel-reconstruction models. One
search did not turn up a weighting of the loss by unexplained change, or
either idea applied to a JEPA. That is one search, not a literature review.

