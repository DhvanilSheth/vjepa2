# V-JEPA 2.1 — Codebase Map

> **Purpose of this document.** The repo at `/Users/dhvanilsheth/Downloads/Personal/vjepa2`
> is Meta FAIR's official PyTorch codebase for **V-JEPA 2**, **V-JEPA 2-AC**, and the
> latest **V-JEPA 2.1**. This map isolates *only the files relevant to V-JEPA 2.1* and
> separates them from the V-JEPA 2 / V-JEPA 2-AC "noise". Every claim below was traced to
> source (`file:line`) or to the V-JEPA 2.1 release commit `45d025f`.
>
> Companion file: [`TODO.md`](./TODO.md) (exploration progress tracker).

---

## 1. TL;DR — What V-JEPA 2.1 is

V-JEPA 2.1 (release `0.0.2`, 2026-03-16, commit `45d025f` "V-JEPA 2.1 (#130)") is a **new
self-supervised pre-training recipe** layered on the same JEPA encoder/predictor skeleton.
Its goal: learn **high-quality, temporally-consistent *dense* features** (good per-patch
representations), not just a strong global embedding.

Paper: *"V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"*
(Mur-Labadia et al., arXiv:2603.14482, 2026).

The recipe rests on four ideas (from the README), all visible in the code:

| # | Innovation | One-line meaning | Where it lives in code |
|---|------------|------------------|------------------------|
| 1 | **Dense Predictive Loss** | The SSL loss is applied to **all** tokens — both masked *targets* and visible *context* — not only the masked region. | `train.py` `predict_all`→`return_all_tokens`; `loss_pred + λ·loss_context` |
| 2 | **Deep Self-Supervision** | The loss is applied at **multiple intermediate encoder layers**, not just the final layer. | `levels_predictor=4`; encoder `out_layers_distillation`/`hierarchical_layers`; predictor multi-head projection |
| 3 | **Multi-Modal Tokenizers** | A single model ingests **both images and video**, with modality-aware tokenization & embeddings. | `img_temporal_dim_size`, `modality_embedding`, `img_rank_ratio`, dual `PatchEmbed3D` |
| 4 | **Model & Data Scaling** | Four sizes B/L/g/G; B & L are **distilled** from the 2B teacher; large mixed image+video corpus. | hub `vjepa2_1_*` factories; `teacher_embed_dim`; `*_dist_vitG` checkpoints |

Plus a fifth that shows up in the **cooldown** phase only: an optional **Gram-matrix feature
loss** (`computing_gram_loss`, `gram_loss_weight`, encoder `gram_mode=...`).

---

## 2. The file map — KEEP vs NOISE

### 2A. ✅ KEEP — NEW, V-JEPA-2.1-only files (the core surface)

These exist *only* because of V-JEPA 2.1. This is the heart of what to read.

```
app/vjepa_2_1/                              # ← the V-JEPA 2.1 training package (self-contained models)
├── train.py            (835 L)  main pre-training loop + dense/deep/multimodal loss
├── utils.py            (368 L)  init_video_model / init_opt / load_checkpoint / normalize_nested
├── wrappers.py         (115 L)  MultiSeqWrapper, PredictorMultiSeqWrapper (multi-FPC, multi-mask)
├── transforms.py       (139 L)  VideoTransform — handles PIL images (T=1) AND video tensors
└── models/                                 # NOTE: v2.1 *forks* its own model code here (diverges from src/models)
    ├── vision_transformer.py (608 L)  ViT encoder w/ RoPE + hierarchical (deep-sup) outputs + dual patch-embed
    ├── predictor.py          (302 L)  predictor w/ mask tokens + return_all_tokens (dense) + teacher out_embed_dim
    └── utils/
        ├── modules.py        (544 L)  Block, Attention, RoPEAttention, MLP/SwiGLU, Lambda_LinearWarmupHold
        ├── patch_embed.py     (72 L)  PatchEmbed (2D), PatchEmbed3D (video, tubelet), AudioPatchEmbed
        ├── pos_embs.py        (95 L)  2D/3D sincos pos-embeds (used when use_rope=False)
        └── masks_dist.py      (77 L)  compute_mask_distance — distance-weighting for the dense context loss

configs/train_2_1/                          # ← V-JEPA 2.1 pre-training recipes (4 sizes × 2 phases)
├── vitb16/{pretrain-256px-16f, cooldown-256px-64f}.yaml
├── vitl16/{pretrain-256px-16f, cooldown-256px-64f}.yaml
├── vitg16/{pretrain-256px-16f, cooldown-256px-64f}.yaml   ⚠ collides w/ vitG16 on macOS (see §8)
└── vitG16/{pretrain-256px-16f, cooldown-256px-64f}.yaml

configs/eval_2_1/                           # ← V-JEPA 2.1 frozen-eval recipes (4 sizes × 7 benchmarks)
├── vitb-384/  ├── vitl-384/  ├── vitg-384/  └── vitG-384/   ⚠ vitg-384/vitG-384 collide on macOS
   each contains: coin · diving48 · ek100 · in1k · jester · k400 · ssv2 .yaml

assets/architecture_vjepa2_1.jpg            # v2.1 architecture figure
assets/teaser_screenshot_5dice.png          # PCA dense-feature teaser
assets/bars_teaser_tikz-1.png               # benchmark bar chart
```

### 2B. ✅ KEEP — SHARED files that V-JEPA 2.1 *modified* (needed to load/run 2.1)

Commit `45d025f` touched these shared files. **Essential** ones are required to load/run 2.1;
the rest are compatibility tweaks or cleanups.

**Essential to run/load V-JEPA 2.1:**
- `src/hub/backbones.py` — adds the 4 hub factories `vjepa2_1_vit_{base,large,giant,gigantic}_384`, a `_make_vjepa2_1_model()` loader (imports from `app.vjepa_2_1.models`), `teacher_embed_dim=1664`, per-model `checkpoint_key`, and an `ARCH_NAME_MAP` for the `*_dist_vitG_384` checkpoints.
- `hubconf.py` — exposes those 4 entrypoints to `torch.hub`.
- `src/models/predictor.py` — adds `out_embed_dim` (derive from `teacher_embed_dim`) so the predictor can project to a teacher of *different* width → enables **distillation** (B/L from G).
- `evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py` — `use_v2_1` flag switches imports to the v2.1 model modules; handles `teacher_embed_dim`, `n_output_distillation`, hierarchical predictor outputs.

**Supporting (correctness/compat):**
- `src/models/vision_transformer.py` — `mpl_ratio`→`mlp_ratio` typo fix for giant/gigantic builders.
- `src/utils/wrappers.py` — exposes `self.embed_dim` on `MultiSeqWrapper`.
- `src/masks/multiseq_multiblock3d.py` — `MaskCollator` now also handles image (fpc=1) batches.
- `evals/action_anticipation_frozen/eval.py` — scaler-restore bugfix (`zip(scaler, …)`).

**Incidental (cleanup / version / paths — safe to skim):**
- `setup.py` (0.0.1→0.0.2), `src/datasets/imagenet1k.py` + `data_manager.py` + `evals/image_classification_frozen/eval.py` (drop unused `image_folder` arg), `src/datasets/video_dataset.py` + `src/datasets/utils/weighted_sampler.py` (formatting), `configs/eval/{vitg-384,vitl}/in1k.yaml` (path cleanup).

### 2C. ⚙️ SHARED — used by V-JEPA 2.1 at runtime but *unchanged* by it

V-JEPA 2.1's `train.py` reuses these `src/` modules as-is (dependencies, not noise, but not 2.1-specific):
- `src/datasets/data_manager.py` (`init_data`), `src/datasets/video_dataset.py`, `src/datasets/imagenet1k.py`
- `src/masks/multiseq_multiblock3d.py` (`MaskCollator`), `src/masks/utils.py` (`apply_masks`)
- `src/utils/distributed.py`, `src/utils/logging.py`, `src/utils/schedulers.py`, `src/utils/tensors.py`
- `app/main.py`, `app/main_distributed.py`, `app/scaffold.py` (generic launchers — dispatch on `app:` key)
- `evals/main.py`, `evals/main_distributed.py`, `evals/scaffold.py` and the per-task `evals/*_frozen/` harnesses (driven by `configs/eval_2_1/*`)

### 2D. ❌ NOISE — V-JEPA 2 / V-JEPA 2-AC only (ignore for 2.1)

- `app/vjepa/` — original V-JEPA 2 pre-training package.
- `app/vjepa_droid/` — V-JEPA 2-**AC** (action-conditioned robot world model).
- `src/models/ac_predictor.py` — action-conditioned predictor (2-AC only).
- `configs/train/` (vitg16/vith16/vitl16, incl. `droid-*`), `configs/inference/`, `configs/eval/` (the non-`_2_1` evals).
- `notebooks/` (energy-landscape / robot-planning demos), `assets/vjepa2-*.png`, `assets/flowchart.png`.
- `src/models/{vision_transformer,predictor,attentive_pooler}.py` are the **shared** copies used by hub + evals; V-JEPA 2.1 *training* uses its own forked copies under `app/vjepa_2_1/models/` instead.

> **Key structural insight:** V-JEPA 2.1 **forked the model definitions** into `app/vjepa_2_1/models/`
> rather than editing `src/models/`. So for *training* read `app/vjepa_2_1/models/*`; the `src/models/*`
> copies matter only for **hub loading** and **frozen evals**.

---

## 3. Architecture deep-dive

### 3.1 Encoder — `app/vjepa_2_1/models/vision_transformer.py`
- Class `VisionTransformer` (`:20`). Factories: `vit_base` (768/12/12), `vit_large` (1024/24/16),
  `vit_giant` (1408/40/16), `vit_giant_xformers` (1408/40/22), `vit_gigantic` (1664/48/16), each with
  `*_rope` variants. `cls_token` is unused (dense-feature focus).
- **Deep self-supervision:** `n_output_distillation` selects intermediate layers via `hierarchical_layers`
  / `out_layers_distillation` (e.g. depth-12 → `[2,5,8,11]`), LayerNorm'd per layer (`norms_block`). In
  training mode the encoder returns the **concatenation** of those intermediate features
  (`torch.cat(hier, dim=2)`, shape `[B,N,embed_dim·#layers]`); at inference it returns just the final norm.
- **Multi-modal tokenizer:** primary `PatchEmbed3D` (`tubelet_size=2`) for video; when
  `img_temporal_dim_size` is set, a second `patch_embed_img` (`PatchEmbed3D`, `tubelet_size=1`) handles
  single-frame images. `modality_embedding=True` adds learnable `img_mod_embed` / `video_mod_embed`.
- **Positions:** `use_rope=True` (default in all 2.1 configs) → rotary embeddings in `RoPEAttention`,
  decomposed over (depth, height, width); `interpolate_rope=True` rescales for the 256→384 res change.
  When `use_rope=False`, falls back to sincos from `pos_embs.py`.

### 3.2 Predictor — `app/vjepa_2_1/models/predictor.py`
- Class `VisionTransformerPredictor` (`:19`), factory `vit_predictor`.
- Input is the encoder's **concatenated hierarchical** features (`embed_dim·#layers`) → `predictor_embed`.
- **Dense loss enabler:** `return_all_tokens=True` makes the predictor emit predictions for **both** the
  masked targets (`predictor_proj`) **and** the visible context (`predictor_proj_context`). Learnable
  `mask_tokens` (count `num_mask_tokens`) replace masked slots.
- **Distillation enabler:** `out_embed_dim` (derived from `teacher_embed_dim=1664`, the 2B ViT-G width)
  lets a small student predictor regress a large teacher's features.

### 3.3 Supporting modules
- `modules.py`: `Block` (norm→attn→MLP), `Attention` + `RoPEAttention` (SDPA, register tokens,
  causal option), `MLP`/`SwiGLUFFN`, `DropPath`, and `Lambda_LinearWarmupHold` (schedules the dense
  context-loss weight λ).
- `masks_dist.py`: `compute_mask_distance()` computes per-token distance weights so the dense **context**
  loss can be spatially weighted (`weight_distance_loss`, `offset_context_loss`).

---

## 4. Training pipeline — `app/vjepa_2_1/train.py`

**Entrypoint:** `python -m app.main --fname configs/train_2_1/<size>/<phase>.yaml` (local) or
`app.main_distributed` (SLURM). The `app: vjepa_2_1` key in the config routes the launcher here.

**Flow:** build encoder+predictor (`init_video_model`) → EMA copy `target_encoder` (`copy.deepcopy`) →
`init_opt` (AdamW/RAdamW + WarmupCosine|LinearDecay LR + Cosine WD + GradScaler) → `init_data`
(video + optional image streams) → `MaskCollator` → per-step:

1. `forward_target(clips)` — EMA encoder, no grad → per-layer LayerNorm'd hierarchical targets `h`.
2. `forward_context(clips)` — online encoder on context tokens → predictor → `(z_pred, z_context)`.
3. **Loss:** `loss = loss_pred(z_pred,h on masks_pred)` **+** `λ_t · loss_context(z_context,h on masks_enc)`
   when `predict_all`. λ scheduled by `Lambda_LinearWarmupHold`; context term optionally distance-weighted.
   `loss_fn` reduces `mean(|z−h|^loss_exp)/loss_exp` (configs use `loss_exp=1.0` → L1) summed over all
   hierarchical layers and masks.
4. Backward (bf16 autocast) → step → **EMA update** `θ_k ← m·θ_k + (1−m)·θ_q` (momentum schedule).

**Multi-modal split:** the first `img_rank_ratio` fraction of ranks process **images**
(crop 512/`tubelet=1`, λ=`lambda_value_img`); the rest process **video** (crop 256/`tubelet=2`,
λ=`lambda_value_vid`). The predictor receives `mod="image"|"video"`.

**Key imports** — local: `models.utils.masks_dist`, `models.utils.modules`, `transforms`, `utils`.
Shared `src/`: `datasets.data_manager.init_data`, `masks.multiseq_multiblock3d.MaskCollator`,
`masks.utils.apply_masks`, `utils.{distributed,logging}`.

---

## 5. Configs / recipe (from `configs/train_2_1/*`)

Common to all sizes: `patch_size=16`, `crop_size=256` (train), `tubelet_size=2`, `use_rope=true`,
`use_mask_tokens=true`, `modality_embedding=true`, `img_temporal_dim_size=1`, `uniform_power=true`,
`interpolate_rope=true`, `use_activation_checkpointing=true`, `pred_embed_dim=384`, `levels_predictor=4`.
Video data = `k710 + ssv2 + howto` (weights `0.335/0.10/0.565`), `fps=4`; image data mixed in at
`rank_ratio=0.5`. Masking = two block strategies (8 small blocks @ scale .15, 2 large @ .70) for video;
10-block image mask. Optim: `lr=6e-4`, `wd=0.04`, `ema=[0.99925, 0.99925]`, `dtype=bfloat16`.

| Size dir | `model_name` | `pred_depth` | params | checkpoint | notes |
|----------|--------------|-------------|--------|-----------|-------|
| `vitb16` | `vit_base`   | 12 | 80M  | `vjepa2_1_vitb_dist_vitG_384.pt` | **distilled** from ViT-G |
| `vitl16` | `vit_large`  | 24 | 300M | `vjepa2_1_vitl_dist_vitG_384.pt` | **distilled** from ViT-G |
| `vitg16` | `vit_giant`  | 24 | 1B   | `vjepa2_1_vitg_384.pt`           | self-supervised teacher |
| `vitG16` | `vit_giant_xformers` | 24 | 2B | `vjepa2_1_vitG_384.pt`     | self-supervised teacher (the distillation teacher, `teacher_embed_dim=1664`) |

**Pretrain vs Cooldown (two-phase recipe):**

| Aspect | `pretrain-256px-16f` | `cooldown-256px-64f` |
|--------|----------------------|----------------------|
| Frames/clip (fpc) | 16 | **64** |
| Epochs | 1000 | 40 |
| Warmup / schedule | 40-epoch warmup, `final_lr=6e-4` | `is_anneal=true`, resumes pretrain ckpt, `final_lr=1e-6` |
| `weight_distance_loss` | **true** | false |
| Extra | `reg_coeff` | optional **Gram loss** (`computing_gram_loss`, `gram_loss_weight=10`, `gram_ckpt`) |
| Batch (video, vitb) | 48 | 24 (halved; bigger compute footprint) |

---

## 6. Frozen-eval suite (`configs/eval_2_1/*`)

Backbone frozen; an **attentive-probe** (16 blocks/heads) is trained with a hyperparameter sweep
(`multihead_kwargs`), bf16, 20 epochs, **resolution 384**. Seven benchmarks per size:

| Benchmark | Task | `eval_name` / module |
|-----------|------|----------------------|
| `in1k`    | image classification | `image_classification_frozen` / `modelcustom.vit_encoder` |
| `k400`    | video classification | `video_classification_frozen` / `vit_encoder_multiclip` |
| `ssv2`    | temporal video classification | `video_classification_frozen` |
| `coin` / `diving48` / `jester` | video classification | `video_classification_frozen` |
| `ek100`   | action anticipation | `action_anticipation_frozen` / `vit_encoder_predictor_concat_ar` (uses **predictor**, `use_v2_1: true`, `teacher_embed_dim=1664`, `num_mask_tokens=8`, `return_all_tokens=true`) |

Checkpoint-key convention in eval configs: distilled students (**B/L**) load `checkpoint_key: ema_encoder`;
teachers (**g/G**) load `checkpoint_key: target_encoder`. Only `ek100` loads the predictor; all other
evals are encoder-only.

---

## 7. Loading V-JEPA 2.1 (PyTorch Hub)

```python
import torch
# V-JEPA 2.1 backbones (added by commit 45d025f)
m_b = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')      # 80M  (distilled)
m_l = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')     # 300M (distilled)
m_g = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_giant_384')     # 1B
m_G = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')  # 2B
```

---

## 8. ⚠️ Known gotcha on this machine — case-insensitive filesystem collision

This checkout sits on a **case-insensitive** macOS filesystem (verified). Git tracks **four** giant-tier
config dirs whose names differ only by case:

| Git index (4 dirs) | On disk (collapses to 1) |
|--------------------|--------------------------|
| `configs/train_2_1/vitg16` (1B) **+** `vitG16` (2B) | only one physical `vitG16/` survives |
| `configs/eval_2_1/vitg-384` (1B) **+** `vitG-384` (2B) | only one physical `vitG-384/` survives |

Reading either case-path returns the **same** file. So **one of the two "giant" config sets (ViT-g 1B
vs ViT-G 2B) is not accessible locally** — they overwrote each other on checkout. The hub factories
(`vjepa2_1_vit_giant_384` vs `vjepa2_1_vit_gigantic_384`) and checkpoints (`vjepa2_1_vitg_384.pt` vs
`vjepa2_1_vitG_384.pt`) remain the source of truth for the distinction. To recover both locally, check
out the repo on a case-sensitive volume (or `git checkout` into a case-sensitive sparse image).

---

## 9. Quick reference — "read these, in this order"

1. `app/vjepa_2_1/train.py` — the recipe end-to-end (loss = §1 innovations 1–3).
2. `app/vjepa_2_1/models/vision_transformer.py` — encoder, deep-sup hierarchical outputs, dual tokenizer.
3. `app/vjepa_2_1/models/predictor.py` — `return_all_tokens` (dense) + `out_embed_dim` (distillation).
4. `app/vjepa_2_1/utils.py` — `init_video_model` / `init_opt` wire the flags above to the configs.
5. `configs/train_2_1/vitb16/pretrain-256px-16f.yaml` — a complete recipe to read top-to-bottom.
6. `src/hub/backbones.py` — how the 4 released checkpoints are loaded.
