# V-JEPA 2.1 — Codebase Mapping TO-DO / Progress Tracker

> Goal: Isolate the files relevant to **V-JEPA 2.1** (latest), cut the V-JEPA 2 noise,
> and produce a clean map (see [`README.md`](./README.md) in this folder).

Legend: ✅ done · 🔄 in progress · ⬜ not started

## Phase 0 — Orientation
- ✅ List full repo file tree
- ✅ Read `CHANGELOG.md` (v2.1 = release `0.0.2`, commit `45d025f`, dated 2026-03-16)
- ✅ Pull exact v2.1 file change-set from git (`git show --stat 45d025f` → 67 files)
- ✅ Read README diff introduced by v2.1 (the official narrative + checkpoint table)
- ✅ Identify the 4 v2.1 innovations (Dense Loss, Deep SSL, MM Tokenizers, Scaling) + Gram loss

## Phase 1 — Deep read: v2.1 NEW code
- ✅ `app/vjepa_2_1/models/*` — encoder (hierarchical/RoPE/dual-tokenizer), predictor (return_all_tokens, out_embed_dim), modules, embeds, masks_dist
- ✅ `app/vjepa_2_1/train.py` + `utils.py` + `wrappers.py` + `transforms.py` — full pipeline + loss math grounded to `train.py:679-703`
- ✅ `configs/train_2_1/*` — pretrain + cooldown recipes (vitb/l/g/G 16) compared
- ✅ `configs/eval_2_1/*` — 4 sizes × 7 benchmarks frozen-eval recipes

## Phase 2 — Deep read: v2.1 MODIFIED shared code
- ✅ `src/hub/backbones.py` — 4 new `vjepa2_1_*` factories + `_make_vjepa2_1_model` + ARCH_NAME_MAP
- ✅ `src/models/{predictor,vision_transformer}.py` diffs (out_embed_dim; mlp_ratio typo fix)
- ✅ `src/masks/multiseq_multiblock3d.py` diff (image-batch support)
- ✅ `src/datasets/*` diffs (image_folder removal, formatting)
- ✅ `src/utils/wrappers.py`, `hubconf.py`, `setup.py` diffs
- ✅ `evals/*` diffs touched by v2.1 (action_anticipation use_v2_1 path; img-cls cleanup; scaler bugfix)
- ✅ Classified essential vs supporting vs incidental

## Phase 3 — Synthesis
- ✅ Write the v2.1 file-map (KEEP-new / KEEP-modified / SHARED-deps / NOISE) → README §2
- ✅ Write the architecture / data-flow summary → README §3–4
- ✅ Document each of the 4 innovations → which file implements it → README §1 table + §3
- ✅ Document how to run v2.1 (train + eval + hub entrypoints) → README §4,6,7
- ✅ Final review pass / verify claims against source (grep-grounded key identifiers)

## Findings / notes
- ✅ **v2.1 forks its model code** into `app/vjepa_2_1/models/` (diverges from `src/models/`);
  `src/models/*` matter only for hub-loading + frozen evals.
- ✅ **B & L are distilled** from the 2B ViT-G teacher (`*_dist_vitG` ckpts; `teacher_embed_dim=1664`;
  students load `ema_encoder`, teachers load `target_encoder`).
- ✅ **Two-phase recipe:** pretrain (16f, 1000ep, warmup) → cooldown (64f, 40ep, anneal, optional Gram loss).
- ⚠️ **Case-insensitive FS gotcha (this machine):** git has 4 giant-tier config dirs
  (`vitg16`+`vitG16`, `vitg-384`+`vitG-384`) that collapse to one on disk — one "giant" config set
  is shadowed locally. See README §8. (Not a code bug; a checkout artifact on macOS.)

## Status: MAPPING COMPLETE — now in bottom-up DEEP-READ phase
All v2.1-relevant files identified, read, and mapped. Deliverables: `README.md` (map) + this tracker.
Optional follow-ups if desired: (a) recover both giant configs via a case-sensitive volume;
(b) line-by-line diff of `app/vjepa_2_1/models/*` vs `src/models/*` to quantify the fork.

---

## Phase 4 — Bottom-up learning walkthrough  → [`WALKTHROUGH.md`](./WALKTHROUGH.md)
Goal: understand the code from the literal basics up to `train.py`, with examples.

- ✅ §1 `models/utils/patch_embed.py` — pixels → tokens (Conv2d/Conv3d patch embed, multimodal hook)
- ✅ §2 `models/utils/pos_embs.py` — sincos 1D/2D/3D positional embeddings (uniform_power)
- ✅ §3 `models/utils/modules.py` — transformer engine
  - ✅ 3.1 MLP / **SwiGLUFFN** (detailed: SiLU, GLU gating, 2/3 param trick) / DropPath
  - ✅ 3.2 Attention (Q/K/V, multi-head, SDPA)
  - ✅ 3.3 RoPE (rotate_queries_or_keys, 3D factorized, mask-as-positions, interpolate_rope)
  - ✅ 3.4 Block (pre-norm residual, config swaps)
  - ✅ 3.5 CrossAttention + Lambda_LinearWarmupHold (λ for dense loss)
- ⬜ §4 `models/utils/masks_dist.py` — compute_mask_distance (distance-weighting the dense loss)  ← NEXT
- ⬜ §5 `models/vision_transformer.py` — assemble the encoder (hierarchical/deep-sup outputs, dual tokenizer)
- ⬜ §6 `models/predictor.py` — return_all_tokens, out_embed_dim (distillation)
- ⬜ §7 glue: `wrappers.py`, `transforms.py`, `utils.py`
- ⬜ §8 `train.py` — the full recipe (dense loss, deep supervision, multimodal, EMA)
